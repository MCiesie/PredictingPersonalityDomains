import json
import torch


def pad_dataset(data_dict, pad_value=0.0):
    text_feats = [torch.as_tensor(txt, dtype=torch.float32) for txt in data_dict["text"]]
    audio_feats = [torch.as_tensor(aud, dtype=torch.float32) for aud in data_dict["audio"]]
    labels = [torch.as_tensor(lab, dtype=torch.float32) for lab in data_dict["labels"]]
    metadata = data_dict["metadata"]

    max_turns_text = max(txt.shape[0] for txt in text_feats)
    max_turns_audio = max(aud.shape[0] for aud in audio_feats)

    # Use the overall max turns for padding all modalities the same length
    max_turns = max(max_turns_text, max_turns_audio)

    padded_text = []
    padded_audio = []
    padded_labels = []
    masks = []


    for txt, aud, lab, meta in zip(text_feats, audio_feats, labels, metadata):
        num_turns_txt = txt.shape[0]
        num_turns_aud = aud.shape[0]

        # pad text & audio
        padded_text.append(torch.nn.functional.pad(txt, (0, 0, 0, max_turns - num_turns_txt), value=pad_value))
        padded_audio.append(torch.nn.functional.pad(aud, (0, 0, 0, max_turns - num_turns_aud), value=pad_value))

        # pad labels (with -1)
        padded_labels.append(torch.nn.functional.pad(lab, (0, max_turns - lab.shape[0]), value=-1))

        # mask: 1 for real turns, 0 for padded
        mask = torch.zeros(max_turns, dtype=torch.bool)
        mask[:lab.shape[0]] = 1
        masks.append(mask)

    return {
        "text": torch.stack(padded_text),
        "audio": torch.stack(padded_audio),
        "labels": torch.stack(padded_labels),
        "mask": torch.stack(masks),
    }

if __name__ == "__main__":
    all_data = torch.load("processed_dataset.pt", weights_only=False)

    with open(f"./metadata.json", "w") as f:
        json.dump(all_data["metadata"], f, indent=2)

    padded_data = pad_dataset(all_data)
    torch.save(padded_data, "./training_dataset.pt")

