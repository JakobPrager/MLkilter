#CustomLoss file

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ordinal Cross-Entropy Loss
def ordinal_cross_entropy(true_grade, predicted_grade):
    """
    Computes ordinal cross-entropy loss.
    Args:
        true_grade (Tensor): Ground truth grades (integer class labels).
        predicted_grade (Tensor): Predicted logits for each class.
    Returns:
        Tensor: Ordinal cross-entropy loss.
    """
    ce_loss = nn.CrossEntropyLoss()
    return ce_loss(predicted_grade, true_grade)

# Grade Distance Loss
def grade_distance_loss(true_grade, predicted_grade):
    """
    Computes the grade distance loss.
    Args:
        true_grade (Tensor): Ground truth grades (integer class labels).
        predicted_grade (Tensor): Predicted logits for each class.
    Returns:
        Tensor: Grade distance loss.
    """
    # Get the predicted class index (argmax of logits)
    predicted_grade_idx = torch.argmax(predicted_grade, dim=1)
    
    # Calculate the absolute difference between true and predicted grades
    distance = torch.abs(true_grade - predicted_grade_idx)
    
    # Apply squared penalty to the distance
    return torch.mean(distance.float() ** 2)

# Combined Loss
def combined_loss(true_grade, predicted_grade):
    """
    Computes the combined loss as a weighted sum of ordinal cross-entropy and grade distance loss.
    Args:
        true_grade (Tensor): Ground truth grades (integer class labels).
        predicted_grade (Tensor): Predicted logits for each class.
    Returns:
        Tensor: Combined loss.
    """
    # Ordinal Cross-Entropy
    ord_loss = ordinal_cross_entropy(true_grade, predicted_grade)
    
    # Grade Distance Loss
    dist_loss = grade_distance_loss(true_grade, predicted_grade)
    
    # Combine losses (adjust the weight of dist_loss as needed)
    return ord_loss + 0.5 * dist_loss
