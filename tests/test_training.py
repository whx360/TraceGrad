import tempfile
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from contextlib import redirect_stdout
import io
from tracegrad.nn import Classifier
from tracegrad.training import classification_metrics,save_checkpoint,load_checkpoint,predict_logits,train_model


class TrainingTests(unittest.TestCase):
    def test_metrics_handcalculation(self):
        logits=np.array([[3.,0.,0.],[0.,3.,0.],[0.,3.,0.],[0.,0.,3.]])
        labels=np.array([0,1,2,2])
        got=classification_metrics(logits,labels)
        self.assertEqual(got['accuracy'],.75)
        np.testing.assert_allclose(got['per_class_f1'],[1.,2/3,2/3])
        self.assertAlmostEqual(got['macro_f1'],7/9)
        self.assertEqual(got['confusion_matrix'],[[1,0,0],[0,1,0],[0,1,1]])

    def test_macro_counts_missing_classes(self):
        got=classification_metrics(np.array([[1.,0.,0.]]),np.array([0]))
        self.assertAlmostEqual(got['macro_f1'],1/3)

    def test_large_finite_cross_entropy_mean(self):
        result=classification_metrics(np.array([[0.,-1e308],[0.,-1e308]]),np.array([1,1]))
        self.assertTrue(np.isfinite(result['cross_entropy']))
        self.assertEqual(result['cross_entropy'],1e308)

    def test_training_rejects_fractional_integer_config(self):
        with tempfile.TemporaryDirectory() as directory:
            for key in ['epochs','batch_size','seed','hidden_dim','classes']:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    train_model({}, {key:.9}, Path(directory)/'run',verbose=False)

    def test_epoch_log_distinguishes_current_and_best(self):
        x=np.array([[1.,0.],[0.,1.]],dtype=np.float32); y=np.array([0,1])
        data={key:value for s in ['train','val','test'] for key,value in [('x_'+s,x),('y_'+s,y)]}
        metrics=[{'cross_entropy':.1,'accuracy':.9},{'cross_entropy':.2,'accuracy':.8},
                 {'cross_entropy':.1,'accuracy':.9},{'cross_entropy':.1,'accuracy':.9}]
        capture=io.StringIO()
        with tempfile.TemporaryDirectory() as directory,patch('tracegrad.training.evaluate',side_effect=metrics),redirect_stdout(capture):
            train_model(data,{'classes':2,'hidden_dim':0,'epochs':2,'batch_size':2},Path(directory)/'run')
        self.assertIn('valCE=0.2000 bestValCE=0.1000 valAcc=0.8000',capture.getvalue())

    def test_metric_invalid_inputs(self):
        with self.assertRaises(ValueError):
            classification_metrics(np.ones((2,3)),np.array([0.,1.]))
        with self.assertRaises(ValueError):
            classification_metrics(np.ones((2,3)),np.array([0,3]))

    def test_checkpoint_roundtrip(self):
        model=Classifier(input_dim=2,hidden_dim=3,classes=2,seed=5,dtype=np.float64)
        x=np.array([[1.,2.],[-1.,.4]])
        meta={'model':{'input_dim':2,'hidden_dim':3,'classes':2,'dtype':'float64'}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz'
            save_checkpoint(path,model,meta)
            other,recovered=load_checkpoint(path)
            self.assertEqual(recovered,meta)
            np.testing.assert_array_equal(predict_logits(model,x),predict_logits(other,x))

    def test_real_train_export_cycle_synthetic_only(self):
        # Synthetic integration fixture, not a real-data/generalization result.
        x=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=np.float32)
        y=np.array([0,0,1,1])
        data={key:value for split in ['train','val','test'] for key,value in [('x_'+split,x),('y_'+split,y)]}
        cfg={'hidden_dim':0,'classes':2,'epochs':20,'batch_size':4,'lr':.1,'seed':9,'protocol':'synthetic_test'}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'
            result=train_model(data,cfg,out,verbose=False)
            self.assertEqual(result['status'],'completed')
            self.assertGreaterEqual(result['train']['accuracy'],.99)
            self.assertTrue((out/'predictions.npz').exists())
            model,_=load_checkpoint(out/'model.npz')
            np.testing.assert_array_equal(predict_logits(model,x).argmax(axis=1),y)
            with self.assertRaises(FileExistsError):
                train_model(data,cfg,out,verbose=False)


if __name__=='__main__': unittest.main()
