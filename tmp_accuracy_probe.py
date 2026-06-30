from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from scripts.accuracy_push_helpers import build_loss, build_model, load_preprocessed_dataset, make_loader


def main() -> None:
    preproc = Path(r"c:\Users\leahs\OneDrive\Documents\Stanford - Durmus Lab\Jupyter Notebook Code\ramanbiolib\outputs\preprocessed\accuracy_push_v7_multiscale_e60")
    data = load_preprocessed_dataset(preproc)
    X_train = data.get("X_train_channels", data["X_train"])
    X_test = data.get("X_test_channels", data["X_test"])
    y_train = data["y_train"]
    y_test = data["y_test"]
    class_names = data["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train_arr = np.asarray(X_train, dtype=np.float32)
    input_channels = 1 if x_train_arr.ndim == 2 else int(x_train_arr.shape[1])

    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=len(class_names)).astype(np.float32)
    class_weights = np.clip(counts.sum() / np.clip(counts, 1.0, None), 1.0, None)
    class_weights = class_weights / np.mean(class_weights)

    model = build_model(
        "multiscale",
        input_len=int(x_train_arr.shape[-1]),
        n_classes=len(class_names),
        input_channels=input_channels,
        base_channels=64,
        conv_dropout=0.15,
        dense_dropout=0.15,
        use_batchnorm=True,
    ).to(device)
    criterion = build_loss(
        "focal",
        label_smoothing=0.0,
        focal_gamma=3.0,
        class_weights=class_weights,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_loader = make_loader(X_train, y_train, batch_size=128, balanced_sampler=True, shuffle=True)
    test_loader = make_loader(X_test, y_test, batch_size=128, balanced_sampler=False, shuffle=False)

    print(f"device={device} train_shape={x_train_arr.shape} test_shape={np.asarray(X_test).shape}", flush=True)
    for epoch in range(1, 11):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0 or epoch == 1:
            model.eval()
            preds = []
            labels = []
            with torch.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb.to(device))
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                    labels.append(yb.numpy())
            preds_arr = np.concatenate(preds)
            labels_arr = np.concatenate(labels)
            print(
                f"epoch={epoch:02d} acc={accuracy_score(labels_arr, preds_arr):.4f} "
                f"macro_f1={f1_score(labels_arr, preds_arr, average='macro', zero_division=0):.4f}",
                flush=True,
            )

    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            labels.append(yb.numpy())
    preds_arr = np.concatenate(preds)
    labels_arr = np.concatenate(labels)
    print(f"final_acc={accuracy_score(labels_arr, preds_arr):.4f}", flush=True)
    print(f"final_macro_f1={f1_score(labels_arr, preds_arr, average='macro', zero_division=0):.4f}", flush=True)


if __name__ == "__main__":
    main()
