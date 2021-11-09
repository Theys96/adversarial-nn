import torch
import math

def saliency_map(J, t, space, size, width):
    S = [(0, 0)] * size
    for p in space:
        alpha = J[t, p // width, p % width].item()
        beta = 0
        for i in range(J.size(0)):
            if not i == t:
                beta += J[i, p // width, p % width].item()
        S[p] = (alpha, beta)
    return S

def clip(orig, val, eps):
    return min([1, orig + eps, max([0, orig - eps, val])])

def jsmaM(original_image_tensor, actual_class, predictor, max_dist, perturbation, epsilon):
    img_tensor = original_image_tensor.clone()
    img_tensor = img_tensor.reshape(28,28)

    min_val = torch.min(img_tensor.reshape(784))
    max_val = torch.max(img_tensor.reshape(784))

    img_tensor = torch.sub(img_tensor, min_val)
    img_tensor = torch.div(img_tensor, max_val - min_val)

    img_tensor = img_tensor.reshape(1,1,28,28)

    img_size = img_tensor.size(2) * img_tensor.size(3)
    width = img_tensor.size(3)
    search_space = list(range(img_size))
    i = 0
    max_iter = math.floor((img_size * max_dist) / (200))
    chosen_pixel_1 = -1
    chosen_pixel_2 = -1
    modifier = 0
    prediction = predictor(img_tensor)
    eta = [0] * img_size

    while prediction.argmax().item() == actual_class and i < max_iter and len(search_space) >= 2:
        max = 0
        J = torch.autograd.functional.jacobian(predictor, img_tensor)[0, :, 0, 0, :, :]
        S = [saliency_map(J, target, search_space, img_size, width) for target in range(10)]
        
        for t in range(10):
            for pixel1 in search_space:
                for pixel2 in search_space:
                    if pixel1 == pixel2:
                        continue
                    
                    alpha = S[t][pixel1][0] + S[t][pixel2][0]
                    beta = S[t][pixel1][1] + S[t][pixel2][1]

                    if -alpha * beta > max:
                        chosen_pixel_1 = pixel1
                        chosen_pixel_2 = pixel2
                        max = -alpha * beta
                        modifier = (-1 if t == actual_class else 1) * math.copysign(1, alpha) * perturbation

        if max == 0:
            break
        
        new1 = clip(
            original_image_tensor[0, 0, chosen_pixel_1 // width, chosen_pixel_1 % width].item(),
            img_tensor[0, 0, chosen_pixel_1 // width, chosen_pixel_1 % width].item() + modifier,
            epsilon
        )
        diff1 = abs(new1 - img_tensor[0, 0, chosen_pixel_1 // width, chosen_pixel_1 % width])
        img_tensor[0, 0, chosen_pixel_1 // width, chosen_pixel_1 % width] = new1
        
        new2 = clip(
            original_image_tensor[0, 0, chosen_pixel_2 // width, chosen_pixel_2 % width].item(),
            img_tensor[0, 0, chosen_pixel_2 // width, chosen_pixel_2 % width].item() + modifier,
            epsilon
        )
        diff2 = abs(new2 - img_tensor[0, 0, chosen_pixel_2 // width, chosen_pixel_2 % width])
        img_tensor[0, 0, chosen_pixel_2 // width, chosen_pixel_2 % width] = new2

        val = img_tensor[0, 0, chosen_pixel_1 // width, chosen_pixel_1 % width]
        if val <= 0 or val >= 1 or diff1 < 1e-06 or eta[chosen_pixel_1] == -1 * modifier:
            search_space.remove(chosen_pixel_1)
    
        val = img_tensor[0, 0, chosen_pixel_2 // width, chosen_pixel_2 % width]
        if val == 0 or val == 1 or diff2 < 1e-06 or eta[chosen_pixel_2] == modifier:
            search_space.remove(chosen_pixel_2)
        
        eta[chosen_pixel_1] = modifier
        eta[chosen_pixel_2] = modifier
        prediction = predictor(img_tensor)

        topPredictions = torch.topk(prediction, 2).indices[0]
        closestIndex = topPredictions[1].item() if prediction.argmax() == actual_class else topPredictions[0].item()
        print(f'Closest attack: {closestIndex} at {prediction[0, closestIndex]}%')

        i += 1
    return img_tensor