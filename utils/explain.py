# utils/explain.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import LayerIntegratedGradients

class AttributeVisualizer:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.model.to(device)

    def get_attention_attributes(self, input_ids, attention_mask):
        """
        Extracts averaged attention weights from the last layer.
        Returns: [seq_len] importance scores.
        """
        with torch.no_grad():
            _, attentions = self.model(
                input_ids.to(self.device), 
                attention_mask.to(self.device), 
                output_attentions=True
            )
        
        # attentions is a tuple of layers, take the last one
        last_layer_attn = attentions[-1] # [B, heads, seq, seq]
        
        # Average across heads
        avg_attn = last_layer_attn.mean(dim=1).squeeze(0) # [seq, seq]
        
        # Sum across rows to get importance per token
        token_importance = avg_attn.sum(dim=0).cpu().numpy()
        
        return token_importance

    def get_gradient_attributes(self, input_ids, attention_mask):
        """
        Uses Captum's Integrated Gradients to attribute the prediction to input tokens.
        Returns: [seq_len] importance scores.
        """
        # LayerIntegratedGradients attributes the output to the input (input_ids)
        # by leveraging the gradients at the specified layer (BERT embeddings).

        lig = LayerIntegratedGradients(self.model, self.model.encoder.bert.embeddings)
        
        # Integrated Gradients requires a baseline (usually all zeros or padding)
        baseline = torch.zeros_like(input_ids).to(self.device)
        
        # We need a wrapper because the model's forward takes (ids, mask)
        # and LIG wants to attribute to a specific input index or keyword arg.
        
        attributions, delta = lig.attribute(
            inputs=input_ids.to(self.device),
            baselines=baseline,
            additional_forward_args=(attention_mask.to(self.device),),
            n_steps=12,
            return_convergence_delta=True
        )
        
        # Sum across embedding dimensions [B, seq, dim] -> [seq]
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions.detach().cpu().numpy()
        
        # Normalize
        if np.linalg.norm(attributions) > 0:
            attributions = attributions / np.linalg.norm(attributions)
            
        return attributions

    def plot_explanation(self, tokens, attributes, title="Token Importance", save_path=None):
        """
        Plots a bar chart of token importance for better readability.
        """
        plt.figure(figsize=(14, 6))
        
        # Use a bar plot instead of a heatmap for much better text spacing
        colors = plt.cm.YlOrRd(attributes / (attributes.max() + 1e-8))
        
        bars = plt.bar(range(len(tokens)), attributes, color=colors, edgecolor='gray', alpha=0.8)
        
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right', fontsize=9)
        plt.ylabel("Importance Score")
        plt.title(title, fontsize=14, pad=20)
        
        # Highlight top 3 tokens
        top_indices = np.argsort(attributes)[-3:]
        for idx in top_indices:
            bars[idx].set_edgecolor('black')
            bars[idx].set_linewidth(2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Explanation saved to {save_path}")
        plt.show()

def explain_main(model, tokenizer, text, device="cpu", save_dir="result/xai"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = AttributeVisualizer(model, tokenizer, device)
    
    enc = tokenizer(
        text, 
        max_length=64, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Remove padding tokens from visualization for clarity
    mask_indices = attention_mask[0] == 1
    tokens = [t for i, t in enumerate(tokens) if mask_indices[i]]
    
    # 1. Attention Map
    attn_attr = visualizer.get_attention_attributes(input_ids, attention_mask)
    attn_attr = attn_attr[mask_indices.cpu().numpy()]
    visualizer.plot_explanation(
        tokens, attn_attr, 
        title="Attention Map Importance", 
        save_path=os.path.join(save_dir, "attention_map.png")
    )
    
    # 2. Integrated Gradients
    try:
        grad_attr = visualizer.get_gradient_attributes(input_ids, attention_mask)
        grad_attr = grad_attr[mask_indices.cpu().numpy()]
        visualizer.plot_explanation(
            tokens, grad_attr, 
            title="Integrated Gradients Importance", 
            save_path=os.path.join(save_dir, "integrated_gradients.png")
        )
    except Exception as e:
        print(f"Could not compute Integrated Gradients: {e}")
