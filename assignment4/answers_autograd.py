"""Runnable PyTorch examples for Assignment 4, Questions 6–9."""

import torch


def show_backward_error(output):
    try:
        output.backward()
    except RuntimeError as error:
        print(type(error).__name__ + ":", error)


def main():
    print("Q6")
    x = torch.tensor([2.0], requires_grad=True)
    (x**2 + 3*x).backward()
    print("x.grad =", x.grad)
    x = torch.tensor([2.0], requires_grad=False)
    show_backward_error(x**2 + 3*x)
    print("default requires_grad =", torch.tensor([2.0]).requires_grad)

    print("\nQ7")
    for track_weights in (False, True):
        x = torch.tensor([2.0], requires_grad=True)
        w = torch.tensor([1.0, 3.0], requires_grad=track_weights)
        y = w[0] * x**2 + w[1] * x
        y.backward()
        print("x.grad =", x.grad, "| w.grad =", w.grad)

    print("\nQ8")
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 3
    z = y.detach()
    show_backward_error(z * 2)
    z = y.clone()
    (z * 2).backward()
    print("x.grad =", x.grad)

    print("\nQ9")
    x = torch.tensor([1.0], requires_grad=True)
    (x * 2).backward()
    print("After first backward: x.grad =", x.grad)
    (x * 3).backward()
    print("After second backward: x.grad =", x.grad)
    x.grad = None
    (x * 3).backward()
    print("After clearing and recomputing: x.grad =", x.grad)


if __name__ == "__main__":
    main()
