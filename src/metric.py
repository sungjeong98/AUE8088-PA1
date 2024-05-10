from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.add_state("confusion_matrix", default=torch.zeros((num_classes, num_classes)), dist_reduce_fx="sum")

    def update(self, preds, target):

        preds = preds.argmax(dim=1)
        for batch in range(target.shape[0]):
            self.confusion_matrix[preds[batch], target[batch]] += 1

    def compute(self):
        precision = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=0) + 1e-8)
        recall = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1) + 1e-8)
        f1_score = (2*precision*recall) / (precision + recall + 1e-8)
        return f1_score.tolist()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = preds.argmax(dim=1)
       
        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape , ("prediction and target have different shape")

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum() 

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
