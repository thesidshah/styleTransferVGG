from torch.autograd import Variable
import os
import torch
from image_operations import gram_matrix, prepare_img, save_and_maybe_display
from model import prepare_model
from parametersTune import build_loss
from torch.optim import LBFGS

def neural_style_transfer(config):
    '''
    The main Neural Style Transfer method.
    '''
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    
    init_img = content_img
    
    optimizing_img = Variable(init_img, requires_grad=True)
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)
    print(f'Using VGG19 in the optimization procedure.')
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]
    num_of_iterations = 1000
    
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, _ = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            print(f'iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}')
            save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations)

        cnt += 1
        return total_loss
        
    optimizer.step(closure)
    
    # def predict(neural_net):
    #     content_images_dir = os.path.join(default_resource_dir, 'content-images')
    #     for content_file_name in os.listdir(content_images_dir):
    #         content_img_path = os.path.join(config['content_images_dir'], content_file_name)
    #         content_img = prepare_img(content_img_path, config['height'], device)
    #         output = neural_net(content_img)
    #         out_img = output.squeeze(axis=0).to('cpu').detach().numpy()
    #         out_img = np.moveaxis(out_img, 0, 2)
    #         img_format = config['img_format']
    #         out_img_name = str(content_file_name).zfill(img_format[0]) + img_format[1] 
    #         dump_img = np.copy(out_img)
    #         dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    #         dump_img = np.clip(dump_img, 0, 255).astype('uint8')
    #         cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])


    # predict(neural_net=neural_net)
    
    return dump_path


PATH = ''
CONTENT_IMAGE = 'c1.jpg'
STYLE_IMAGE = 's1.jpg'


default_resource_dir = os.path.join(PATH, 'data')
content_images_dir = os.path.join(default_resource_dir, 'content-images')
style_images_dir = os.path.join(default_resource_dir, 'style-images')
output_img_dir = os.path.join(default_resource_dir, 'output-images')
img_format = (4, '.jpg')



# import os
# assign directory
# directory = 'files'
 
# iterate over files in
# that directory

for contentfilename in list(['c8.jpg']):
    for stylefilename in list(['s1.jpg','s3.jpg','s4.jpg','s6.jpg','s9.jpg','s12.jpg','s8.jpg','s23.jpg']):
        optimization_config = {'content_img_name': contentfilename, 'style_img_name': stylefilename, 'height': 400, 'content_weight': 100000.0, 'style_weight': 30000.0, 'tv_weight': 1.0}
        optimization_config['content_images_dir'] = content_images_dir
        optimization_config['style_images_dir'] = style_images_dir
        optimization_config['output_img_dir'] = output_img_dir
        optimization_config['img_format'] = img_format

        results_path = neural_style_transfer(optimization_config)

