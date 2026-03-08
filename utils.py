"""
Shared utilities for visual brain encoding project.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_paths(base_dir):
    """Get all data paths."""
    train_dir = os.path.join(base_dir, 'training_data', 'subj02')
    test_dir = os.path.join(base_dir, 'test_data', 'subj02')
    
    return {
        'train_img': os.path.join(train_dir, 'training_split', 'training_images'),
        'train_fmri': os.path.join(train_dir, 'training_split', 'training_fmri'),
        'test_img': os.path.join(train_dir, 'test_split', 'test_images'),
        'test_fmri': os.path.join(test_dir, 'test_split', 'test_fmri'),
        'roi': os.path.join(train_dir, 'roi_masks')
    }


def load_fmri_data(paths):
    """Load fMRI data for both hemispheres."""
    lh_train = np.load(os.path.join(paths['train_fmri'], 'lh_training_fmri.npy'))
    rh_train = np.load(os.path.join(paths['train_fmri'], 'rh_training_fmri.npy'))
    lh_test = np.load(os.path.join(paths['test_fmri'], 'lh_test_fmri.npy'))
    rh_test = np.load(os.path.join(paths['test_fmri'], 'rh_test_fmri.npy'))
    
    print(f"Training fMRI - LH: {lh_train.shape}, RH: {rh_train.shape}")
    print(f"Test fMRI - LH: {lh_test.shape}, RH: {rh_test.shape}")
    
    return {'lh_train': lh_train, 'rh_train': rh_train, 
            'lh_test': lh_test, 'rh_test': rh_test}


def load_roi_masks(paths):
    """Load ROI masks for place-selective regions (floc-places)."""
    lh_roi = np.load(os.path.join(paths['roi'], 'lh.floc-places_challenge_space.npy'))
    rh_roi = np.load(os.path.join(paths['roi'], 'rh.floc-places_challenge_space.npy'))
    roi_mapping = np.load(os.path.join(paths['roi'], 'mapping_floc-places.npy'), allow_pickle=True).item()
    
    print(f"ROI masks - LH: {lh_roi.shape}, RH: {rh_roi.shape}")
    print(f"ROI mapping: {roi_mapping}")
    
    return {'lh': lh_roi, 'rh': rh_roi, 'mapping': roi_mapping}


def get_roi_vertices(roi_masks, fmri_data, n_vertices=10):
    """Select random vertices for each ROI (OPA, PPA, RSC)."""
    roi_data = {}
    mapping = roi_masks['mapping']
    
    # Mapping is {idx: name}, iterate correctly
    for roi_idx, roi_name in mapping.items():
        if roi_name == 'Unknown' or roi_idx == 0:
            continue  # Skip unknown/background
            
        lh_vertices = np.where(roi_masks['lh'] == roi_idx)[0]
        rh_vertices = np.where(roi_masks['rh'] == roi_idx)[0]
        
        print(f"\n{roi_name}: LH has {len(lh_vertices)} vertices, RH has {len(rh_vertices)} vertices")
        
        selected_lh = []
        selected_rh = []
        
        # Try to get vertices from LH first, then RH
        if len(lh_vertices) >= n_vertices:
            selected_lh = np.random.choice(lh_vertices, n_vertices, replace=False)
        elif len(lh_vertices) > 0:
            selected_lh = lh_vertices
            remaining = n_vertices - len(lh_vertices)
            if len(rh_vertices) >= remaining:
                selected_rh = np.random.choice(rh_vertices, remaining, replace=False)
            elif len(rh_vertices) > 0:
                selected_rh = rh_vertices
        elif len(rh_vertices) >= n_vertices:
            # No LH vertices, use RH only
            selected_rh = np.random.choice(rh_vertices, n_vertices, replace=False)
        elif len(rh_vertices) > 0:
            selected_rh = rh_vertices
        
        train_responses = []
        test_responses = []
        
        if len(selected_lh) > 0:
            train_responses.append(fmri_data['lh_train'][:, selected_lh])
            test_responses.append(fmri_data['lh_test'][:, selected_lh])
        if len(selected_rh) > 0:
            train_responses.append(fmri_data['rh_train'][:, selected_rh])
            test_responses.append(fmri_data['rh_test'][:, selected_rh])
        
        if train_responses:
            roi_data[roi_name] = {
                'train': np.concatenate(train_responses, axis=1) if len(train_responses) > 1 else train_responses[0],
                'test': np.concatenate(test_responses, axis=1) if len(test_responses) > 1 else test_responses[0],
                'lh_idx': selected_lh,
                'rh_idx': selected_rh
            }
            print(f"  Selected {len(selected_lh)} LH + {len(selected_rh)} RH vertices")
    
    return roi_data


def get_image_paths(img_dir):
    """Get sorted list of image paths."""
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    return [os.path.join(img_dir, img) for img in images]


def train_encoding_model(X_train, y_train, X_test, y_test, alpha=1000):
    """Train ridge regression encoding model and evaluate."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    correlations = []
    for v in range(y_test.shape[1]):
        r, _ = pearsonr(y_test[:, v], y_pred[:, v])
        correlations.append(r if not np.isnan(r) else 0)
    
    return np.array(correlations), np.mean(correlations), model


def compute_rdm(features):
    """Compute Representational Dissimilarity Matrix."""
    return squareform(pdist(features, metric='correlation'))


def compute_rsa(rdm1, rdm2):
    """Compute RSA (correlation between upper triangles of RDMs)."""
    from scipy.stats import spearmanr
    upper1 = rdm1[np.triu_indices(rdm1.shape[0], k=1)]
    upper2 = rdm2[np.triu_indices(rdm2.shape[0], k=1)]
    rsa, _ = spearmanr(upper1, upper2)
    return rsa


def linear_cka(X, Y):
    """Compute linear Centered Kernel Alignment."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    K = X @ X.T
    L = Y @ Y.T
    
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    
    hsic_kl = np.sum(K_c * L_c)
    hsic_kk = np.sum(K_c * K_c)
    hsic_ll = np.sum(L_c * L_c)
    
    return hsic_kl / (np.sqrt(hsic_kk * hsic_ll) + 1e-10)
