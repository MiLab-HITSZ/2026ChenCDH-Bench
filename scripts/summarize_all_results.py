import json
import os
import glob

def summarize_results():
    summary_files = glob.glob("result/*/summary.json")
    results = []

    for file_path in summary_files:
        model_name = os.path.basename(os.path.dirname(file_path))
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 1. Overall results
            overall = data.get("overall", {})
            for task, metrics in overall.items():
                results.append({
                    "Model": model_name,
                    "Task": task,
                    "Level": "Overall",
                    "Category": "All",
                    "CF_Acc": metrics.get("CF_Acc"),
                    "CS_Acc": metrics.get("CS_Acc"),
                    "Gap": metrics.get("Gap"),
                    "CCR": metrics.get("CCR"),
                    "RPD": metrics.get("RPD")
                })
            
            # 2. Category results
            by_category = data.get("by_category", {})
            for task, categories in by_category.items():
                for cat_name, metrics in categories.items():
                    results.append({
                        "Model": model_name,
                        "Task": task,
                        "Level": "Category",
                        "Category": cat_name,
                        "CF_Acc": metrics.get("CF_Acc"),
                        "CS_Acc": metrics.get("CS_Acc"),
                        "Gap": metrics.get("Gap"),
                        "CCR": metrics.get("CCR"),
                        "RPD": metrics.get("RPD")
                    })

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Sort by Task, then Level (Overall -> Category -> Subcategory), then Category, then Model
    level_order = {"Overall": 0, "Category": 1, "Subcategory": 2}
    results.sort(key=lambda x: (x["Task"], level_order.get(x["Level"], 3), x["Category"], x["Model"]))
    
    # Print a markdown table
    header = "| Model | Task | Level | Category | CF_Acc | CS_Acc | Gap | CCR | RPD |"
    separator = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    
    print(header)
    print(separator)
    
    for r in results:
        cf = f"{r['CF_Acc']:.4f}" if r['CF_Acc'] is not None else "N/A"
        cs = f"{r['CS_Acc']:.4f}" if r['CS_Acc'] is not None else "N/A"
        gap = f"{r['Gap']:.4f}" if r['Gap'] is not None else "N/A"
        ccr = f"{r['CCR']:.4f}" if r['CCR'] is not None else "N/A"
        rpd = f"{r['RPD']:.4f}" if r['RPD'] is not None else "N/A"
        print(f"| {r['Model']} | {r['Task']} | {r['Level']} | {r['Category']} | {cf} | {cs} | {gap} | {ccr} | {rpd} |")

if __name__ == "__main__":
    summarize_results()
