import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from data_ops import get_val_dl


ckpt_path = "./model_store/tf_efficientnet_b3.pth"
params = OmegaConf.load("./configs/cosface_tuned")
test_loader = get_val_dl(params, "test")
device = "cuda"
model = torch.load(ckpt_path)["model"]
model = model.to("cuda")
model.eval()

num_best_guess = 5
test_pred_paths, test_pred_names = [], []
list_pred = []
with torch.no_grad():
    for batch_idx, (inputs, paths, data_indices) in enumerate(test_loader):
        inputs  = inputs.to(device)
        outputs = model(inputs)
        _,best_values = torch.sort(outputs, dim=1)
        best_values   = best_values.cpu().numpy()
        best_values_shape  = best_values.shape
        best_value_decoder = label_encoder.inverse_transform(best_values.ravel())
        best_value_decoder = best_value_decoder.reshape(best_values_shape)[:,-num_best_guess:][:,::-1]
        test_pred_paths.append(paths)
        test_pred_names.append(best_value_decoder)
        list_pred.append(outputs.cpu().numpy())

list_pred  = np.vstack(list_pred)
np.save(f"{os.path.dirname(resume)}/list_pred.npy", list_pred)
