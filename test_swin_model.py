# test_swin_model.py
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


def test_swin_model():
    """测试训练好的Swin模型"""

    # 加载类别名称
    try:
        with open('swin_class_names.txt', 'r', encoding='utf-8') as f:
            class_names = [line.strip().split(': ')[1] for line in f.readlines()]
        print(f"加载 {len(class_names)} 个类别")
    except:
        print("无法加载类别名称文件")
        return

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型架构
    import timm
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=len(class_names))

    # 加载训练好的权重
    try:
        checkpoint = torch.load('best_swin_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"模型加载成功，验证准确率: {checkpoint['val_acc']:.2f}%")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 测试单张图像（这里需要您提供测试图像路径）
    test_image_path = input("请输入测试图像路径: ").strip().strip('"')

    try:
        image = Image.open(test_image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        print(f"\n预测结果:")
        print(f"  类别: {class_names[predicted_class]}")
        print(f"  置信度: {confidence:.2%}")
        print(f"  类别索引: {predicted_class}")

        # 显示前3个预测结果
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        print(f"\nTop 3 预测:")
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            print(f"  {i + 1}. {class_names[idx]}: {prob:.2%}")

    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    test_swin_model()