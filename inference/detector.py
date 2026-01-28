# inference/detector.py
import torch


class LogSentryDetector:
    def __init__(self, classifier, retriever, beta=0.68):
        self.classifier = classifier
        self.retriever = retriever
        self.beta = beta

    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            logit = self.classifier(input_ids, attention_mask)
            prob_model = torch.sigmoid(logit).item()

            embedding = self.classifier.encoder(input_ids, attention_mask)
            prob_knn = self.retriever.query(embedding.cpu().numpy()[0])

        final_score = self.beta * prob_model + (1 - self.beta) * prob_knn
        return final_score > 0.5, final_score
