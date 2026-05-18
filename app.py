from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import timm

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD BASE MODELS
# =========================

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load("resnet.pth", map_location=device))
resnet.eval()

densenet = models.densenet121(weights=None)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 2)
densenet.load_state_dict(torch.load("densenet.pth", map_location=device))
densenet.eval()

convnext = timm.create_model("convnext_base", pretrained=False, num_classes=2)
convnext.load_state_dict(torch.load("convnext.pth", map_location=device))
convnext.eval()

effnet = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=2)
effnet.load_state_dict(torch.load("effnet.pth", map_location=device))
effnet.eval()

swin = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=2)
swin.load_state_dict(torch.load("swin.pth", map_location=device))
swin.eval()

models_list = [resnet, densenet, convnext, effnet, swin]

# =========================
# FUSION MODEL
# =========================

class Fusion(nn.Module):

    def __init__(self, models):
        super().__init__()

        self.models = nn.ModuleList(models)

        self.attn = nn.Sequential(
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,5)
        )

    def entropy(self, p):
        return -torch.sum(p * torch.log(p+1e-8), dim=1)

    def forward(self, x):

        outs = []
        ents = []

        for m in self.models:

            o = torch.softmax(m(x), dim=1)

            outs.append(o)
            ents.append(self.entropy(o))

        outs = torch.stack(outs)

        concat = outs.permute(1,0,2).reshape(x.size(0), -1)

        attn = torch.softmax(self.attn(concat), dim=1)

        ents = torch.stack(ents).permute(1,0)

        conf = 1/(ents+1e-6)
        conf = conf / conf.sum(dim=1, keepdim=True)

        w = 0.7*attn + 0.3*conf
        w = w / w.sum(dim=1, keepdim=True)

        w = w.unsqueeze(2)

        outs = outs.permute(1,0,2)

        return (outs * w).sum(dim=1)

fusion = Fusion(models_list).to(device)

fusion.load_state_dict(torch.load("fusion_model.pth", map_location=device))

fusion.eval()

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("PMOS.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = Image.open(file).convert("RGB")

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():

        output = fusion(img)

        pred = torch.argmax(output, dim=1).item()

    result = "PCOS Detected" if pred == 1 else "No PCOS Detected"

    return f"<h1>{result}</h1>"

# =========================

if __name__ == "__main__":
    app.run(debug=True)