import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

def compute_text_embeddings(
    text_data,
    embedding_dir,
    model_name="sentence-transformers/all-distilroberta-v1",
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(embedding_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    with open(text_data, "r", encoding="utf-8") as f:
        all_text_data = json.load(f)

    for session_data in all_text_data:
        if 1 not in session_data["session"] and 8 not in session_data["session"]:
            continue
        
        pid = session_data["PID"]
        session_num = 1 if 1 in session_data["session"] else 8
        turns = session_data["turns"]

        embeddings_list = []

        for turn_text in turns:
            inputs = tokenizer(
                turn_text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # Mean pooling over token embeddings
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                turn_embedding = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings_list.append(turn_embedding[0])

        embeddings_array = np.array(embeddings_list)
        save_path = Path(embedding_dir) / f"{pid}_session{session_num}_embeddings.npy"
        np.save(save_path, embeddings_array)

        print(f"Saved embeddings: {save_path}")

    print("All embeddings computed and saved")


if __name__ == "__main__":
    compute_text_embeddings(
        text_data="./all_data_patient.json",
        embedding_dir="./text_embeddings_patient",
        model_name="sentence-transformers/all-distilroberta-v1"
    )
