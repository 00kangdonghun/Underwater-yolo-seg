import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO

# <model>

# model =YOLO("/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/models/v8/yolov8-seg.yaml").load("/home1/danny472/Underwater/SpikeYOLO/yolov8n-seg.pt")

# model =YOLO("/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/models/v8/yolov8-seg.yaml")

model = YOLO("/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/models/v8/uw-final.yaml")

print(model)

# <train>

#############################################################################################
import sys
sys.path.append("/home1/danny472/Underwater/SpikeYOLO/ultralytics/utils/callbacks")  
from callbacks_uw import on_preprocess_batch
model.add_callback("on_preprocess_batch", on_preprocess_batch)
#############################################################################################

# <dataset>
model.train(data="/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/datasets/UIIS10K.yaml", device="0,1,2,3", epochs=100, project="/home1/danny472/Underwater/SpikeYOLO/runs")  # train the model
# model.train(data="/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/datasets/deposition_302.yaml", device="0,1,2,3", epochs=100, project="/home1/danny472/Underwater/SpikeYOLO/runs")  # train the model
# model.train(data="/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/datasets/marine_262.yaml", device="0,1,2,3", epochs=100, project="/home1/danny472/Underwater/SpikeYOLO/runs")  # train the model
# model.train(data="/home1/danny472/Underwater/SpikeYOLO/ultralytics/cfg/datasets/trash_295.yaml", device="0,1,2,3", epochs=100, project="/home1/danny472/Underwater/SpikeYOLO/runs")  # train the model


# python train.py > ./STDOUT/${LOG_PREFIX}.out 2> ./STDERR/${LOG_PREFIX}.err

# <TEST>
# model = YOLO('/home1/danny472/Underwater/SpikeYOLO/runs/UIIS10K/uw-final/train/weights/best.pt')  # load a pretrained model (recommended for training)

# <predict>
# yolo detect predict model=/home1/danny472/Underwater/SpikeYOLO/runs/UIIS10K/uw-final/train/weights/best.pt source='/home1/danny472/Underwater/dataset/UIIS10K/images/multiclass_test/test_0002.jpg'
