import tensorflow as tf
from tensorflow.keras import backend as K

def get_asymmetric_ttf_loss(alpha=5.0, imminent_threshold=0.2, imminent_weight=3.0):
    """
    Returns a custom asymmetric loss function for Time-to-Failure (TTF) regression.
    
    OVERESTIMATING remaining life is dangerous (predicting 30 days when it's 5).
    UNDERESTIMATING is safer (predicting 5 days when it's 30).
    
    Args:
        alpha: Penalty multiplier for overestimation (predictions > true values).
        imminent_threshold: Normalized threshold below which a failure is "imminent".
        imminent_weight: Weight multiplier for imminent failures.
    """
    
    def asymmetric_loss(y_true, y_pred):
        # y_true and y_pred should be normalized TTF (0.0 to 1.0)
        error = y_pred - y_true
        
        # Standard squared error
        squared_error = tf.square(error)
        
        # Heavy penalty for overestimation (error > 0)
        loss = tf.where(error > 0, alpha * squared_error, squared_error)
        
        # Extra weight if the true failure is imminent
        weights = tf.where(y_true < imminent_threshold, imminent_weight, 1.0)
        
        return K.mean(loss * weights)
        
    return asymmetric_loss

def register_custom_objects():
    """Register custom objects for model loading."""
    tf.keras.utils.get_custom_objects().update({
        'asymmetric_loss': get_asymmetric_ttf_loss()
    })
