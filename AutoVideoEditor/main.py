import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPVisionModel
from tqdm import tqdm
import os
import argparse

from utils import contrastive_loss, load_dataloader
from VideoTextDataset import create_dataset
from model import AutoClipModel

def get_args():
    parser = argparse.ArgumentParser(description="Training Args")

    # File config
    parser.add_argument('--json_path', type=str, help='path to the precomputed JSON file')
    parser.add_argument('--preprocess_file', type=str, default='process_dataloader.pt', help='path to save the preprocessed dataloader')
    parser.add_argument('--out_dir', type=str, default='', help='output directory for model weights and logs')

    # Data process config
    parser.add_argument('--n_frames', type=int, default=8, help='number of frames to sample from each clip')
    parser.add_argument('--frame_size', type=int, default=512, help='size of frames (e.g., 224 for 224x224)')
    parser.add_argument('--target_fps', type=int, default=10, help='the processing fps')

    # Training config
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the training on')

    # Model config
    parser.add_argument('--t', type=float, default=0.07, help='temperature for projection')
    parser.add_argument('--projection_dim', type=int, default=512, help='dimension of the projection space')

    return parser.parse_args()

def main():
    config = get_args()
    # make necessary dirs
    os.makedirs(config.out_dir, exist_ok=True)

    # Data Preprocessing
    save_path = os.path.join(config.out_dir, config.preprocess_file)
    if not os.path.exists(save_path):
        # Create and save the dataset
        dataset = create_dataset(config.json_path, config.n_frames, config.frame_size, config.target_fps)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        torch.save(dataloader, save_path)
        print(f"Data preprocessing completed and saved to {save_path}.")
    else:
        # Load the saved DataLoader
        dataloader = load_dataloader(save_path)

    # Model Setup
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(config.device)
    vision_input_dim = vision_model.config.hidden_size
    text_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").text_model.to(config.device)
    text_input_dim = text_model.config.hidden_size

    # Initialize AutoClipModel
    model = AutoClipModel(vision_model, vision_input_dim, text_model, text_input_dim, config.projection_dim, config.t).to(config.device)
    # check model
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_loss = 100
    tolerance = 5
    count = 0

    # Training Loop
    for epoch in range(config.epochs):
        model.train()
        for batch in tqdm(dataloader, desc='Training:'):
            # video input
            video_frames_data = batch['video'].to(config.device)
            # text input
            caption_input_ids = batch['caption_input_ids'].to(config.device)
            caption_attention_mask = batch['caption_attention_mask'].to(config.device)
            text_input = {
                'input_ids': caption_input_ids,
                'attention_mask': caption_attention_mask
            }

            # forward
            logit_video, logit_text = model(video_frames_data, text_input)

            # contrastive loss
            loss = contrastive_loss(logit_video, logit_text, config.device, config.t)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch: {epoch + 1} || Loss: {loss.item()}")
        if loss.item() < best_loss:
            count = 0
            torch.save(model.state_dict(), os.path.join(config.out_dir, "best_model.pt"))
            print("Best model weight saved!")
            best_loss = loss.item()
        else:
            count += 1
            if count > tolerance:
                torch.save(model.state_dict(), os.path.join(config.out_dir, "last_model.pt"))
                print("No improvement, early stopping.")
                exit()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(config.out_dir, "last_model.pt"))
    print("Training completed and model weights saved.")

if __name__ == "__main__":
    main()
