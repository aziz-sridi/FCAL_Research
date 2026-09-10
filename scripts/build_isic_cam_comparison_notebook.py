from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build(output: str | Path = "notebooks/isic_all_models_cam_comparison.ipynb") -> Path:
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [
        markdown(
            "# ISIC CAM comparison: CE vs FCAL V1 vs FCAL V2\n\n"
            "This notebook compares all six trained architectures using their best-validation checkpoints. "
            "For each architecture, the same eight held-out test lesions (one per class) are shown beside "
            "the CE, V1, and V2 predicted-class CAMs. The green/white contour is the lesion mask.\n\n"
            "The examples are selected automatically from the test set: within each class, choose a valid "
            "non-full-image mask with the greatest prediction disagreement across all 18 checkpoints. "
            "This makes failures visible without changing the test data or selecting separately for a favored method."
        ),
        code(
            "from pathlib import Path\n"
            "import gc, json\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import torch\n"
            "from torch.nn import functional as F\n"
            "from IPython.display import display\n\n"
            "from fcal_study.config import load_config\n"
            "from fcal_study.data import IMAGENET_MEAN, IMAGENET_STD, IsicDataset\n"
            "from fcal_study.losses import leakage\n"
            "from fcal_study.models import create_model\n\n"
            "PROJECT = Path.cwd().resolve()\n"
            "if not (PROJECT / 'configs' / 'isic_resnet50_v1_v2.yaml').is_file():\n"
            "    candidates = [p for p in [PROJECT, *PROJECT.parents] if (p / 'configs' / 'isic_resnet50_v1_v2.yaml').is_file()]\n"
            "    if not candidates: raise FileNotFoundError('Run from the FCAL repository root')\n"
            "    PROJECT = candidates[0]\n"
            "base_config = load_config(PROJECT / 'configs' / 'isic_resnet50_v1_v2.yaml')\n"
            "manifest = json.loads((Path(base_config['data']['split_dir']) / 'manifest.json').read_text(encoding='utf-8'))\n"
            "class_names = manifest['class_order']\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "mean = torch.tensor(IMAGENET_MEAN)[:, None, None]\n"
            "std = torch.tensor(IMAGENET_STD)[:, None, None]\n"
            "FIGURE_ROOT = PROJECT / 'reports' / 'isic_cam_galleries'\n"
            "FIGURE_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "plt.rcParams.update({'figure.dpi': 105, 'axes.titlesize': 8})\n"
            "print('Device:', device)\n"
            "print('Test classes:', ', '.join(class_names))"
        ),
        markdown("## 1. Checkpoints included"),
        code(
            "architectures = {\n"
            "    'resnet50': ('ResNet-50', 'resnet50'),\n"
            "    'densenet121': ('DenseNet-121', 'densenet121'),\n"
            "    'efficientnet-b0': ('EfficientNet-B0', 'efficientnet-b0'),\n"
            "    'vit-tiny-p16-avg': ('ViT-Tiny/16 avg', 'vit-tiny-p16-avg'),\n"
            "    'vit-small-p16-avg': ('ViT-Small/16 avg', 'vit-small-p16-avg'),\n"
            "    'vit-base-p16-avg': ('ViT-Base/16 avg', 'vit-base-p16-avg'),\n"
            "}\n"
            "methods = {\n"
            "    'CE': 'ce-seed42',\n"
            "    'V1': 'fcal-v1-seed42',\n"
            "    'V2': 'fcal-v2-seed42',\n"
            "}\n"
            "run_root = PROJECT / 'outputs' / 'isic'\n"
            "checkpoint_rows = []\n"
            "for key, (display_name, run_prefix) in architectures.items():\n"
            "    for method, suffix in methods.items():\n"
            "        run_name = f'{run_prefix}-{suffix}'\n"
            "        checkpoint = run_root / run_name / 'best-validation.pt'\n"
            "        evaluation = run_root / run_name / 'evaluation' / 'clean.npz'\n"
            "        checkpoint_rows.append({'architecture_key': key, 'architecture': display_name, 'method': method, 'run': run_name, 'checkpoint': checkpoint, 'evaluation': evaluation, 'ready': checkpoint.is_file() and evaluation.is_file()})\n"
            "checkpoint_table = pd.DataFrame(checkpoint_rows)\n"
            "display(checkpoint_table[['architecture','method','run','ready']])\n"
            "assert checkpoint_table.ready.all(), 'At least one checkpoint/evaluation is missing'"
        ),
        markdown(
            "## 2. Full 100-image test metrics\n\n"
            "These metrics use every held-out test image, not only the gallery examples. `CAM inside lesion` "
            "is one minus strict leakage for the ground-truth-class CAM. Higher is better. The localization "
            "metric is undefined for a zero CAM, so the zero-map percentage is reported separately."
        ),
        code(
            "report = pd.read_csv(PROJECT / 'reports' / 'isic_v1_v2' / 'primary_metrics.csv')\n"
            "method_map = {'Baseline': 'CE', 'FCAL V1': 'V1', 'FCAL V2': 'V2'}\n"
            "report['method'] = report.method.map(method_map)\n"
            "report['clean_accuracy_percent'] = 100 * report.clean_accuracy\n"
            "report['balanced_accuracy_percent'] = 100 * report.clean_balanced_accuracy\n"
            "report['macro_f1_percent'] = 100 * report.clean_macro_f1\n"
            "report['CAM_inside_lesion_percent'] = 100 * (1 - report.strict_leakage)\n"
            "zero_fractions = []\n"
            "for item in checkpoint_table.itertuples(index=False):\n"
            "    data = np.load(item.evaluation)\n"
            "    zero_fractions.append({'architecture': item.architecture_key, 'method': item.method, 'zero_CAM_percent': 100 * (1 - data['nonzero_map'].mean())})\n"
            "zero_fractions = pd.DataFrame(zero_fractions)\n"
            "metrics = report.merge(zero_fractions, on=['architecture','method'])\n"
            "metric_columns = ['architecture','method','clean_accuracy_percent','balanced_accuracy_percent','macro_f1_percent','CAM_inside_lesion_percent','zero_CAM_percent']\n"
            "display(metrics[metric_columns].round(2).sort_values(['architecture','method']))"
        ),
        markdown(
            "## 3. Select the same eight test lesions for every model\n\n"
            "Selection uses only class label, mask validity, and prediction disagreement across all methods. "
            "It excludes masks covering 95% or more of the image because those contours cannot test localization."
        ),
        code(
            "test_dataset = IsicDataset(base_config, 'test', need_mask=True)\n"
            "test_csv = pd.read_csv(Path(base_config['data']['split_dir']) / 'test.csv')\n"
            "reference_ids = None\n"
            "reference_labels = None\n"
            "all_predictions = []\n"
            "for item in checkpoint_table.itertuples(index=False):\n"
            "    data = np.load(item.evaluation)\n"
            "    ids, labels, predictions = data['image_id'], data['label'], data['prediction']\n"
            "    if reference_ids is None:\n"
            "        reference_ids, reference_labels = ids, labels\n"
            "    else:\n"
            "        assert np.array_equal(ids, reference_ids) and np.array_equal(labels, reference_labels)\n"
            "    all_predictions.append(predictions)\n"
            "all_predictions = np.stack(all_predictions)\n"
            "diversity = np.array([len(set(all_predictions[:, i].tolist())) for i in range(len(reference_ids))])\n"
            "correct_votes = (all_predictions == reference_labels[None,:]).sum(axis=0)\n"
            "mask_fractions = np.array([float(test_dataset[i]['mask'].mean()) for i in range(len(test_dataset))])\n"
            "selected_indices = []\n"
            "for label in range(len(class_names)):\n"
            "    candidates = np.flatnonzero((reference_labels == label) & (mask_fractions < .95))\n"
            "    order = sorted(candidates.tolist(), key=lambda i: (-diversity[i], -min(correct_votes[i], 18-correct_votes[i]), int(reference_ids[i])))\n"
            "    selected_indices.append(order[0])\n"
            "selection = pd.DataFrame({\n"
            "    'dataset_index': selected_indices,\n"
            "    'image_id': reference_ids[selected_indices],\n"
            "    'class': [class_names[x] for x in reference_labels[selected_indices]],\n"
            "    'mask_source': test_csv.iloc[selected_indices].mask_source.to_numpy(),\n"
            "    'mask_area_percent': 100*mask_fractions[selected_indices],\n"
            "    'distinct_predictions_across_18_models': diversity[selected_indices],\n"
            "    'correct_models_out_of_18': correct_votes[selected_indices],\n"
            "})\n"
            "display(selection.round(2))"
        ),
        code(
            "selected_samples = [test_dataset[i] for i in selected_indices]\n"
            "selected_images = torch.stack([sample['image'] for sample in selected_samples])\n"
            "selected_masks = torch.stack([sample['mask'] for sample in selected_samples])\n"
            "selected_labels = torch.tensor([sample['label'] for sample in selected_samples], dtype=torch.long)\n"
            "display_images = (selected_images * std + mean).permute(0,2,3,1).clamp(0,1).numpy()\n"
            "display_masks = selected_masks[:,0].numpy() > .5\n"
            "print('Prepared gallery tensor:', tuple(selected_images.shape))"
        ),
        markdown(
            "## 4. Generate predicted-class CAMs from all 18 checkpoints\n\n"
            "CAMs are computed at each model's native evidence resolution and bilinearly enlarged to 224x224. "
            "Each map is normalized independently for color display. `inside` in the gallery title is the "
            "fraction of positive predicted-class CAM mass falling inside the strict lesion mask."
        ),
        code(
            "cam_results = {}\n"
            "for item in checkpoint_table.itertuples(index=False):\n"
            "    print(f'Running {item.architecture} {item.method}...', flush=True)\n"
            "    checkpoint = torch.load(item.checkpoint, map_location='cpu', weights_only=False)\n"
            "    model_config = checkpoint['config']\n"
            "    model = create_model(model_config, pretrained=False)\n"
            "    model.load_state_dict(checkpoint['model'])\n"
            "    del checkpoint\n"
            "    model.to(device).eval()\n"
            "    images = selected_images.to(device)\n"
            "    labels = selected_labels.to(device)\n"
            "    masks = selected_masks.to(device)\n"
            "    if model_config['training'].get('channels_last', False):\n"
            "        model.to(memory_format=torch.channels_last)\n"
            "        images = images.contiguous(memory_format=torch.channels_last)\n"
            "    amp_mode = model_config['training'].get('amp', 'off')\n"
            "    amp_dtype = torch.bfloat16 if amp_mode == 'bf16' else torch.float16\n"
            "    context = torch.autocast('cuda', dtype=amp_dtype) if device.type == 'cuda' and amp_mode != 'off' else torch.no_grad()\n"
            "    with torch.inference_mode(), context:\n"
            "        logits = model(images, capture_attention=True)\n"
            "        probabilities = logits.float().softmax(1)\n"
            "        predictions = probabilities.argmax(1)\n"
            "        native_cam = model.evaluation_attention(predictions).float()\n"
            "    strict_leakage = leakage(native_cam, masks, 0)\n"
            "    enlarged = F.interpolate(native_cam[:,None], size=(224,224), mode='bilinear', align_corners=False)[:,0]\n"
            "    flat = enlarged.flatten(1)\n"
            "    enlarged = enlarged - flat.min(1).values[:,None,None]\n"
            "    enlarged = enlarged / enlarged.flatten(1).max(1).values[:,None,None].clamp_min(1e-8)\n"
            "    cam_results[(item.architecture_key, item.method)] = {\n"
            "        'cam': (255*enlarged.clamp(0,1)).byte().cpu().numpy(),\n"
            "        'prediction': predictions.cpu().numpy(),\n"
            "        'confidence': probabilities.max(1).values.cpu().numpy(),\n"
            "        'inside': (1-strict_leakage).cpu().numpy(),\n"
            "        'native_size': tuple(native_cam.shape[-2:]),\n"
            "    }\n"
            "    del model, images, labels, masks, logits, probabilities, predictions, native_cam, enlarged\n"
            "    gc.collect()\n"
            "    if device.type == 'cuda': torch.cuda.empty_cache()\n"
            "print('CAM sets generated:', len(cam_results))"
        ),
        markdown(
            "## 5. CAM galleries\n\n"
            "Columns are directly comparable within each architecture. A check mark means the prediction is "
            "correct; a cross means it is wrong. The heatmap explains the predicted class, while the contour "
            "shows the lesion region used by FCAL."
        ),
        code(
            "def cam_gallery(architecture_key, architecture_name):\n"
            "    fig, axes = plt.subplots(len(selected_indices), 4, figsize=(14, 2.65*len(selected_indices)), constrained_layout=True)\n"
            "    for row in range(len(selected_indices)):\n"
            "        axes[row,0].imshow(display_images[row])\n"
            "        axes[row,0].contour(display_masks[row], [.5], colors='lime', linewidths=1.1)\n"
            "        metadata = selection.iloc[row]\n"
            "        axes[row,0].set_title(f\"{metadata['class']} | ID {int(metadata['image_id'])}\\n{metadata['mask_source']} | mask={metadata['mask_area_percent']:.1f}%\")\n"
            "        axes[row,0].axis('off')\n"
            "        for column, method in enumerate(('CE','V1','V2'), start=1):\n"
            "            result = cam_results[(architecture_key, method)]\n"
            "            prediction = int(result['prediction'][row])\n"
            "            correct = prediction == int(selected_labels[row])\n"
            "            axes[row,column].imshow(display_images[row])\n"
            "            axes[row,column].imshow(result['cam'][row]/255, cmap='jet', alpha=.52, vmin=0, vmax=1)\n"
            "            axes[row,column].contour(display_masks[row], [.5], colors='white', linewidths=1.0)\n"
            "            mark = 'correct' if correct else 'WRONG'\n"
            "            axes[row,column].set_title(f\"{method}: {class_names[prediction]} ({mark})\\np={result['confidence'][row]:.2f} | inside={result['inside'][row]:.2f} | native {result['native_size'][0]}x{result['native_size'][1]}\")\n"
            "            axes[row,column].axis('off')\n"
            "    fig.suptitle(f'{architecture_name}: CE vs V1 vs V2 predicted-class CAMs', fontsize=14)\n"
            "    return fig\n\n"
            "for architecture_key, (architecture_name, _) in architectures.items():\n"
            "    figure = cam_gallery(architecture_key, architecture_name)\n"
            "    figure.savefig(FIGURE_ROOT / f'{architecture_key}_ce_v1_v2_cam.png', dpi=140, bbox_inches='tight')\n"
            "    display(figure)\n"
            "    plt.close(figure)"
        ),
        markdown(
            "## 6. Selected-example CAM localization table\n\n"
            "This table averages only the eight displayed lesions and should be treated as a gallery summary, "
            "not as the primary test result. The 100-image table above is the reportable comparison."
        ),
        code(
            "gallery_rows = []\n"
            "for architecture_key, (architecture_name, _) in architectures.items():\n"
            "    for method in ('CE','V1','V2'):\n"
            "        result = cam_results[(architecture_key, method)]\n"
            "        gallery_rows.append({\n"
            "            'architecture': architecture_name, 'method': method,\n"
            "            'gallery_accuracy_percent': 100*np.mean(result['prediction'] == selected_labels.numpy()),\n"
            "            'mean_predicted_CAM_inside_percent': 100*np.mean(result['inside']),\n"
            "            'median_predicted_CAM_inside_percent': 100*np.median(result['inside']),\n"
            "        })\n"
            "gallery_summary = pd.DataFrame(gallery_rows)\n"
            "display(gallery_summary.round(2))"
        ),
        markdown(
            "## Reading the notebook\n\n"
            "A better FCAL model should generally move predicted-class CAM mass inside the white lesion contour "
            "without sacrificing classification. Expect V1 to resemble CE because its three-cell tolerance "
            "collapses on coarse maps. V2 should often be more lesion-focused, but a focused CAM does not guarantee "
            "the correct diagnosis. Full-image or poor-quality masks are audited separately in "
            "`isic_mask_overlay_audit.ipynb`."
        ),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    return output_path


def execute(path: str | Path) -> Path:
    notebook_path = Path(path).resolve()
    notebook = nbf.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbf.write(notebook, notebook_path)
    return notebook_path


if __name__ == "__main__":
    print(execute(build()))
