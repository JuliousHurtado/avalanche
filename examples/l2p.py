import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import LearningToPrompt

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    train_transform = transforms.Compose([
        transforms.Resize(224),    
        transforms.RandomCrop(size=(224, 224), padding=4), 
        # transforms.Resize(28),    
        # transforms.RandomCrop(size=(28, 28), padding=4), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # lambda x: x / 225,
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)) 
        ])

    eval_transform = transforms.Compose([ 
        transforms.Resize(224),
        # transforms.Resize(28),  
        transforms.ToTensor(),
        # lambda x: x / 225,
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)) 
        ])

    # --- BENCHMARK CREATION
    benchmark = SplitCIFAR100(n_experiences=10, return_task_id=False,
            train_transform=train_transform,
            eval_transform=eval_transform)

    # choose some metrics and evaluation method
    loggers = [TextLogger()]

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
           experience=True, stream=True
        ),
        loss_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=loggers,
    )

    cl_strategy = LearningToPrompt(
        model_name="vit_base_patch16_224", #, vit_base_patch32_224 "vit_base_patch16_224", simpleMLP
        criterion=CrossEntropyLoss(),
        train_mb_size=32,
        train_epochs=5,
        num_classes=100,
        eval_mb_size=128,
        lr=0.0075,
        device=device,
        evaluator=eval_plugin,
        use_cls_features=True,
        use_mask=True,
        use_vit=True,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for task_id, experience in enumerate(benchmark.train_stream):
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream[:task_id+1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )

    args = parser.parse_args()
    main(args)
