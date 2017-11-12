import numpy as np

def convnet(I):
    """
    Takes an input image and run it through conv->maxpool->conv->maxpool->conv->maxpool.
    
    Args:
        I: Input image tensor
        
    Returns:
        F: Feature map tensor    
    """
    pass

def defaultgen_legacy( I, F, width, height): 
    """
    Generates an array of default boxes: TESTED
    
    Args:
        I: Input image tensor
        F: Feature map tensor to generate default boxes for
        width: Width of the box
        height: Height of the box
        
    Returns:
        db: Default boxes tensor, flattened
    """
    #number of boxes is the number of pixels in feature map. Default box for each pixel.  
    boxes = np.zeros((F.shape[0], F.shape[1], 6))
    
    for i in range(F.shape[0]): 
        for j in range(F.shape[1]): 
            
            # d depends on the number of max pooling layers
          
            scale = (I.shape[0]/ F.shape[0]) 
            cx  = ((( scale- 1)/ 2 )+ (j*(scale )))
            cy  = (((scale - 1)/ 2 )+ (i*(scale ))) 
            
            boxes[i,j,:] = np.array([cx,cy,width, height, 0, 0])
            
    return boxes.reshape(boxes.shape[0] * boxes.shape[1], 6)

def defaultgen(I, F, width, height):
    scale = I.shape[2] / F.shape[2]
    offset = np.ones((2,F.shape[2],F.shape[3]))
    iterator = np.flip(np.indices((F.shape[2],F.shape[3])), axis=0)
    xys = offset * ((scale-1)/2) + iterator * scale
    xys = np.moveaxis(xys,0,2)
    ws = np.full((F.shape[2],F.shape[3],1), width)
    hs = np.full((F.shape[2],F.shape[3],1), height)
    class_scores = np.zeros((F.shape[2],F.shape[3],2))
    print(xys.shape, ws.shape, hs.shape, class_scores.shape)
    zs = np.dstack([xys, ws, hs, class_scores])
    return zs.reshape((zs.shape[0] * zs.shape[1], 6))

def shiftgen(F):
    """ 
    Generates shifts for each box dimension, relative to the default box width and height
    
    Args:
        F: Feature map from convnet
        
    Returns:
        shifts: Flattened array of shift vector for each box
    
    """
    pass
    

def predict(Bshift, B):
    """
    Predicts absolute box positions from relative shifts: TESTED
    
    Args:
        Bshift: Box tensor
        B: Tensor of default boxes
        
    Returns:
        Predicted_boxes: Tensor of absolute box positions
    """
    
    cx_pred = Bshift[:,0]*B[:,2] + B[:,0]
    cy_pred = Bshift[:,1]*B[:,2] + B[:,1]
    W_pred = np.exp(Bshift[:,2])*B[:,2]
    H_pred = np.exp(Bshift[:,3])*B[:,3]
    class1 = Bshift[:,4]
    class2 = Bshift[:,5]
    Predicted_boxes = np.vstack((cx_pred,cy_pred,W_pred,H_pred, class1, class2))
    Predicted_boxes = np.transpose(Predicted_boxes)
    
    return Predicted_boxes

def center_to_coords(box):
    """ 
    Converts bounding box from (cx, cy, w, h) format to (x1, y1, x2, y2) format
    
    Args:
        box: 6d numpy array
        
    Returns:
        box_transformed: Box in the second format
    
    """
    box_transformed = np.array([box[0]-box[2]/2.0, box[1]+box[3]/2.0, box[0]+box[2]/2.0, box[1]-box[3]/2.0, box[4], box[5]])
    return box_transformed

def iou(boxA, boxB):
    """
    Calculates intersection over union score of two rectangles, none of them rotated.: TESTED
    
    Args:
        boxA: Box1
        boxB: Box2
        
    Returns:
        iou (scalar): IOU score of the two boxes
    """
        # determine the (x, y)-coordinates of the intersection rectangle
        
    boxA = center_to_coords(boxA)
    boxB = center_to_coords(boxB)
    
    xA = max(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxa_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxb_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxa_area + boxb_area - interArea)

    # return the intersection over union value
    return iou

def find_match_indices(pred_boxes, gt): 
    """
    Find matches between ground truth boxes and boxes predicted by network.
    
    Args:
        gt (): Tensor of ground truth boxes
        pred_boxes (): Tensor of predicted boxes
        
    Returns:
        pos_inds: Indices of the connectivity matrix where there is a positive match
        neg_inds: Indices of the connectivity matrix where there is a negative match
    """
    
    pos_iou_score =  np.zeros((gt.shape[0], pred_boxes.shape[0]))
    neg_iou_score = np.ones((gt.shape[0], pred_boxes.shape[0]))
    
    for i in range(gt.shape[0]): 
        for j in range(pred_boxes.shape[0]): 
            iou_score  = iou(gt[i], pred_boxes[j])
            
            if iou_score > 0.5:  
                pos_iou_score[i,j] = iou_score
            else :
                neg_iou_score[i,j] = iou_score
    
    pos_inds  = np.argwhere(pos_iou_score>0.5)
    neg_inds = np.argwhere(neg_iou_score<=0.5)
    
    return  pos_inds, neg_inds

def class_loss(pred_boxes, gt, pos_indices, neg_indices):
    """
    Calculates classification loss between ground truth and predicted boxes generated by the network.
    
    Args:
        pred_boxes: Array of predicted boxes
        gt: Array of ground truth boxes
        pos_indiceis: Positive connectivity matrix
        neg_indices: Negative connectivity matrix
        
    Returns:
        class_loss (scalar): Classification loss
    """
    # predicted boxes classes
    dbclass = pred_boxes[:,4:6]
    
    #ground truth boxes classes
    gtclass = gt[:,4:6]
    
    #exponential of the predicted classes
    exp_dbclass=np.exp(dbclass) 
    
    #sum of the exponential 
    sum_exp_dbclass = np.sum(np.exp(dbclass), axis=1)
    
    db_probs =np.array([exp_dbclass[:,0]/sum_exp_dbclass,exp_dbclass[:,1]/sum_exp_dbclass] )
    db_probs = db_probs.transpose()
    
    pos_loss= 0 
    
    for i in range(pos_inds.shape[0]):
        
        p_ind = pos_inds[i,:]
        p_ind = p_ind[0].astype(np.uint8)
        gt_prob = gtclass[p_ind[0], : ]
        db_prob = db_probs[p_ind[1], : ]
        pos_loss+= np.dot(gt_prob, -np.log(db_prob))
    
    neg_loss= 0     
    for i in range(neg_inds.shape[0]):
        p_ind = neg_inds[i,:]
        p_ind = p_ind.astype(np.uint8)
        gt_prob = gtclass[p_ind[0], : ]
        db_prob = db_probs[p_ind[1], : ]
        neg_loss+= np.dot(gt_prob, -np.log(1-db_prob))
        
    return pos_loss + neg_loss
        
def regr_loss(shifts, gt, pos_indices, neg_indices):
    """
    Calculates regression loss between ground truth and predicted shifts generated by the network.
    
    Args:
        shifts: Array of relative shifts for each box dimension
        gt: Array of ground truths
        pos_indices: Positive connectivity matrix
        neg_indices: Negative connectivity matrix
        
    Returns:
        reg_loss (sclar): Regression loss
    """
    pass
    