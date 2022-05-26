import torch.onnx
from nets.onet import ONet

#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3,48,48, requires_grad=True)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "eye.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['data_input'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'data_input' : {1 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {1 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = ONet()
    # path = "logs/ep027-loss0.010-val_loss0.172.pth"
    path = "logs_mask_close/ep036-loss0.004-val_loss0.142.pth"
    model.load_state_dict(torch.load(path,map_location="cpu"))

    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX()