import requests
from PIL import Image
from io import BytesIO
from model import *
import torch
import numpy as np

class Predictor:
  id_to_class = {
    0: 'Lotus',
    1: 'Tulip',
    2: 'Orchid',
    3: 'Lilly',
    4: 'Sunflower'
  }

  def __init__(self, model_path: str) -> None:
    state = torch.load(model_path)
    model = FlowerCNN()
    model.load_state_dict(state)
    self.device = torch.device(self.device_name)
    self.model = model.to(self.device)
    model.eval()
  
  @property
  def device_name(self) -> str:
    if torch.cuda.is_available():
      return 'cuda'
    if torch.backends.mps.is_available():
      return 'mps'
    return 'cpu'

  def predict(self, image: Image) -> str:
    tensor = self.prepare_tensor(image)
    tensor = tensor.to(self.device)
    with torch.no_grad():
      output = self.model(tensor)
      _, predicted = torch.max(output, 1)
      return self.id_to_class[predicted.item()]
  
  def prepare_tensor(self, image: Image) -> torch.Tensor:
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = image.transpose((2, 0, 1))
    image = torch.Tensor(image).to(torch.float32)
    return image.unsqueeze(0)


def get_image_from_url(url: str) -> Image:
  response = requests.get(url)
  image = Image.open(BytesIO(response.content))
  return image

def get_image_from_user() -> Image:
  while True:
    url = input('Enter an image url: ')
    try: return get_image_from_url(url)
    except Exception: print('Invalid url. Please try again.')

def main() -> None:
  while True:
    image = get_image_from_user()
    predictor = Predictor('./models/model.pth')
    print('Predicting...')
    print(predictor.predict(image))

if __name__ == '__main__':
  main()