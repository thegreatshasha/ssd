import unittest
import numpy as np
from functions import defaultgen, iou, find_match_indices

class TestAvalon(unittest.TestCase):
    
        def setup(self):
            print self._testMethodName
            
        def _test_convnet(self):
            pass # No idea how to test this yet
            
        def _test_conv2d(self):
            pass
            
        def test_defaultgen(self):
            # Ok so what is the logic for testing default gen?
            # We can take a 8*4 image and resize it to a 2x1 feature map. Then we can match the coords as expected
            images = [np.random.random((4,8,3))]
            feature_maps = [np.random.random((1,2,16))]
            true_boxes = [np.array([[1, 1, 20, 10, 0, 0], [5, 1, 20, 10, 0, 0]])]
            # Add more boxes
            
            for I, F, true_box in zip(images, feature_maps, true_boxes):
                boxes = defaultgen(I, F, 20, 10)
                for arr1, arr2 in zip(boxes, true_box):
                    np.testing.assert_array_almost_equal(arr1, arr2)
                    
        def test_iou(self):
            # Checks if the iou score between two boxes is a certain value
            box1 = np.array([0, 0, 2, 2, 0, 0])
            box2 = np.array([1, 1, 2, 2, 0, 0])
            np.testing.assert_almost_equal(iou(box1, box2), 1.0/7, 3, 'IOU not working') # Manually worked out case
            
            box3 = np.array([0.5,0.5,3,3,0,0])
            box4 = np.array([0,0,4,2,0,0])
            np.testing.assert_almost_equal(iou(box3, box4), 0.5454, 3, 'IOU not working')
        
        def _test_shiftgen(self):
            # This is just a linear convolution2d and needs no testing
            pass
        
        def test_predict(self):
            # Transform -> Reverse transform ~ original
            I  = np.ones((8,8))
            F = np.ones((4, 4))
            W = 10 
            H = 10 
            
            def_boxes = defaultgen( I, F, W, H)
            
            Bshift= np.zeros((F.shape[0]*F.shape[1], 6))
            Bshift[:,0:4]= 1
            
            abs_boxes = predict(Bshift, def_boxes)
            
            for box, def_box in zip(abs_boxes, def_boxes):
                cx_rel = (box[0] - def_box[0])/def_box[2] # Check if this is a percentage shift
                cy_rel = (box[1] - def_box[1])/def_box[3] # Check if this is a percentage shift
                cw_rel = np.log(box[2]/def_box[2]) # Log of scaling factor
                ch_rel = np.log(box[3]/def_box[3]) # Log of scaling factor
                
                shift_vector = np.array([cx_rel, cy_rel, cw_rel, ch_rel])
                np.testing.assert_almost_equal(shift_vector, np.ones(4), 3, 'Error in prediction. Transform + Reverse don\'t map to identity')
            pass
        
        def _test_iou(self):
            pass
        
        def test_find_match_indices(self):
            gt = np.array([[0.5,0.5,3,3,0,0]])
            pred_boxes = np.array([[0,0,4,2,0,0], [2.5,2.5,3,3,0,0], [12.5,2.5,3,3,0,0]])
            
            pos_inds, neg_inds = find_match_indices(pred_boxes, gt)
            
            np.testing.assert_array_equal(pos_inds[0], [0, 0], 'Posititve matching not working')
            np.testing.assert_array_equal(neg_inds[0], [0, 1], 'Negative matching not working')
            np.testing.assert_array_equal(neg_inds[1], [0, 2], 'Positive matching not working')
        
        def _test_conf_loss(self):
            pass
        
        def _test_reg_loss(self):
            pass
        
if __name__ == "__main__":
    unittest.main()