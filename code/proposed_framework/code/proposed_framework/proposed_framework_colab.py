# 0) Install dependencies
!pip -q install ultralytics opencv-python numpy pandas matplotlib

# 1) Imports and constants
import math, csv, os, sys, re, shutil
from statistics import mean
from typing import List, Dict

from ultralytics import YOLO
import cv2
import numpy as np

from google.colab import files

# ====== Scenario name and output folder ======
scenario_name = input("\nScenario name: ").strip()
if not scenario_name:
    scenario_name = "unnamed_scenario"

# Make the name safe for folders and ZIP files.
safe_scenario_name = re.sub(r"[^A-Za-z0-9_-]+", "_", scenario_name).strip("_")
if not safe_scenario_name:
    safe_scenario_name = "unnamed_scenario"

OUTPUT_DIR = f"/content/{safe_scenario_name}_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📁 Scenario: {scenario_name}")
print(f"📁 Output folder: {OUTPUT_DIR}")

# ====== Fixed settings ======
EXITS = ["E1", "E6", "E7"]   # Three fixed exits

# Area around each exit (m²)
exit_areas = {
    "E1": 10.3,
    "E6": 7.0,
    "E7": 8.31,
}

WEIGHT_TIME, WEIGHT_RHO, WEIGHT_FEAR = 0.408, 0.388, 0.204
INF_COST = 1e10

# ====== FEAR model ======
FEAR_A, FEAR_B = 10.98, 0.053
FEAR_CONST_DIST, FEAR_CONST_VAL = 20.0, 3.93

def distance_m(p1, p2):
    (x1,y1),(x2,y2) = p1,p2
    return math.hypot(x1-x2, y1-y2)/100.0  # coordinate units -> metres

def fear_func(d_m: float) -> float:
    return FEAR_A*math.exp(-FEAR_B*d_m) if d_m < FEAR_CONST_DIST else FEAR_CONST_VAL

def parse_nodes_line(s: str) -> List[str]:
    return [t.strip() for t in s.replace(",", " ").split() if t.strip()]

def natural_key(name: str):
    digits = "".join(c for c in name if c.isdigit())
    return (name[0] if name else "", int(digits) if digits else -1, name)

# ====== YOLO helper ======
def yolo_count_people_for_exits(
    images_by_exit: Dict[str, List[str]],
    weights: str = "yolov8n.pt",
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = ""    # "", "cpu", or "0" for GPU
) -> Dict[str, float]:
    """
    images_by_exit: {"E1":["/content/e1_1.jpg",...], "E6":[...], "E7":[...]}
    Returns the mean detected person count for each exit.
    """
    model = YOLO(weights)
    out = {}
    for ex, paths in images_by_exit.items():
        counts = []
        for p in paths:
            if not p:
                continue
            if not os.path.isfile(p):
                alt = f"/content/{os.path.basename(p)}"
                p = alt if os.path.isfile(alt) else p
            if not os.path.isfile(p):
                print(f"⚠️ Image not found: {p}")
                continue
            img = cv2.imread(p)
            if img is None:
                print(f"⚠️ Unreadable or corrupted image: {p}")
                continue
            res = model.predict(source=img, imgsz=960, conf=conf, iou=iou, device=device, verbose=False)
            det = res[0]
            if det.boxes is None:
                counts.append(0)
                continue
            cls = det.boxes.cls.cpu().numpy().astype(int)
            counts.append(int((cls == 0).sum()))  # COCO: person=0
        out[ex] = float(mean(counts)) if counts else 0.0
    return out

def upload_images_for_exit(exit_name: str, max_images: int = 3) -> List[str]:
    """
    Opens the file picker for one exit. Select up to three images.
    Uploaded files are stored in /content and their paths are returned.
    """
    print(f"\n📸 Please select up to {max_images} images for exit {exit_name} (multiple files can be selected together).")
    uploaded = files.upload()
    saved_paths = []
    for i, (name, data) in enumerate(uploaded.items()):
        if i >= max_images:
            break
        p = f"/content/{name}"
        saved_paths.append(p)
    if not saved_paths:
        print(f"⚠️ No image was provided for {exit_name}; the initial person count is set to 0.")
    return saved_paths

# 2) Node-coordinate dictionary
nodes = {
    # N nodes
    'N1': (-6393.4642, -5837.4895),
    'N2': (-6156.3168, -5510.9107),
    'N3': (-6699.1348, -5707.5622),
    'N4': (-6872.6571, -5594.8163),
    'N5': (-6640.9351, -5400.5989),
    'N6': (-6776.8067, -5222.1754),
    'N7': (-6480.6678, -5193.4053),
    'N8': (-6668.5502, -5009.5524),
    'N9': (-6170.5941, -6293.9598),
    'N10': (-5970.3017, -6086.1602),
    'N11': (-5811.2319, -6216.2742),
    'N12': (-5752.9347, -5903.2915),
    'N13': (-5611.2376, -6060.7914),
    'N14': (-6092.2823, -4715.975),
    'N15': (-6358.7434, -4421.6826),
    'N16': (-5584.0633, -4115.504),
    'N17': (-6091.4489, -3900.9294),
    'N18': (-5875.5247, -3537.405),
    'N19': (-5510.8978, -3117.0524),
    'N20': (-5172.7122, -3441.9229),
    'N21': (-4577.439, -3996.8748),
    'N22': (-4911.8114, -4323.9531),
    'N23': (-5107.7909, -4524.3265),
    'N24': (-5481.9767, -4647.8111),
    'N25': (-5206.7714, -4888.5816),
    'N26': (-5328.6906, -5550.0445),
    'N27': (-5021.1692, -5830.5581),
    'N28': (-4536.7603, -5188.5493),
    'N29': (-4717.2177, -4893.6782),
    'N30': (-4015.7437, -4593.4913),
    'N31': (-3440.4761, -4007.4353),
    'N32': (-3253.5717, -4475.6791),
    'N33': (-3692.6737, -4922.6509),
    'N34': (-2954.2772, -4180.3196),
    'N35': (-2550.9015, -3771.4134),
    'N36': (-2857.4117, -3437.4922),
    'N37': (-4579.4157, -2848.4519),
    'N38': (-5048.7943, -2666.1033),
    'N39': (-4769.5751, -2389.6727),
    'N40': (-3970.0189, -2250.8233),
    'N41': (-4350.0169, -1969.404),
    'N42': (-3427.3869, -2857.5307),
    'N43': (-4007.1143, -3427.287),
    'N44': (-2916.4006, -2331.201),
    'N45': (-2550.5217, -2246.0833),
    'N46': (-2809.0803, -1970.7575),
    'N47': (-2311.3282, -2860.5839),
    'N48': (-2019.3608, -3195.7403),
    'N49': (-1806.6327, -2234.2323),
    'N50': (-1563.4534, -1764.5384),
    'N51': (-1377.7603, -1810.5842),
    'N52': (-1336.4821, -1625.2575),
    'N53': (-1242.7146, -1288.7277),
    'N54': (-1477.625, -1297.8312),
    'N55': (-1387.3462, -1193.5936),
    'N56': (-1280.8382, -1067.4634),
    'N57': (-1865.1329, -1283.2642),
    'N58': (-2276.9012, -988.7204),
    'N59': (-2450.1279, -809.6344),
    'N60': (-2686.9138, -1332.5029),
    'N61': (-3014.5853, -1092.2865),
    'N62': (-3314.6065, -1866.1394),
    'N63': (-3113.1674, -2550.3006),
    'N64': (-2211.1685, -2211.1685),
    'N65': (-1861.4921, -669.9502),
    'N66': (-1851.6387, -920.7819),
    'N67': (-1738.7808, -811.7534),
    'N68': (-1639.6091, -710.1068),
    'N69': (-3859.1414, -1573.8722),
    'N70': (-3575.9301, -1338.3954),

    # D nodes
    'D1': (-5881.3496, -5970.1923),
    'D2': (-5909.8037, -6143.1586),
    'D3': (-6062.8003, -6178.6589),
    'D4': (-5210.6679, -5670.3479),
    'D5': (-5694.8245, -6011.7521),
    'D7': (-5030.8929, -5249.779),
    'D8': (-6751.453, -5493.2827),
    'D9': (-6537.9657, -5308.6044),
    'D10': (-6585.839, -5123.6539),
    'D11': (-6223.7515, -4658.1178),
    'D12': (-6713.7207, -5331.943),
    'D13': (-5819.6482, -4450.1512),
    'D14': (-5543.6018, -4534.6416),
    'D15': (-5335.1207, -4338.1001),
    'D16': (-4907.3211, -4780.0418),
    'D17': (-5107.6614, -4968.9482),
    'D18': (-5858.2809, -3964.475),
    'D19': (-5773.0659, -3880.6389),
    'D20': (-5265.0026, -3363.593),
    'D21': (-5064.6523, -3526.6951),
    'D22': (-4496.7303, -2950.8402),
    'D23': (-4709.5763, -2820.755),
    'D24': (-4625.5126, -2731.2372),
    'D25': (-3925.9888, -2376.021),
    'D26': (-4100.3014, -2209.9556),
    'D27': (-3564.5472, -1675.4741),
    'D28': (-3483.216, -1596.1554),
    'D29': (-3111.2698, -2093.0361),
    'D30': (-2679.4551, -2547.1306),
    'D31': (-2475.9152, -2340.0945),
    'D32': (-2913.6229, -1900.7037),
    'D33': (-2989.0611, -1625.4584),
    'D34': (-2788.9653, -1228.4898),
    'D35': (-2378.9359, -882.5031),
    'D36': (-2378.9369, -925.2449),
    'D37': (-2070.7665, -850.2548),
    'D38': (-1893.6347, -740.1349),
    'D39': (-1788.5163, -629.179),
    'D40': (-1425.7069, -1324.6394),
    'D41': (-1313.5736, -1210.4873),
    'D42': (-1203.4615, -1100.3752),
    'D43': (-1498.01, -1498.0014),
    'D44': (-1442.5036, -1810.1876),
    'D45': (-2023.9706, -2590.8213),
    'D46': (-2773.0538, -3522.9295),
    'D47': (-3304.0457, -4058.0817),
    'D48': (-3389.262, -4143.6129),
    'D49': (-3931.6176, -4684.83),
    'D50': (-4100.0208, -4496.8659),
    'D51': (-3529.7661, -3920.1839),
    'D52': (-2953.4522, -3346.1571),

    # Exit nodes
    'E1': (-6480.8753, -6137.2182),
    'E2': (-6174.8186, -6001.9911),
    'E3': (-5596.9508, -5620.3954),
    'E4': (-5414.6268, -5438.6321),
    'E5': (-5240.0013, -5262.8881),
    'E6': (-5866.4552, -4698.8141),
    'E7': (-2750.5408, -1586.2544),
    'E8': (-2187.306, -2209.7993),
    'E9': (-2008.8032, -2028.7501),
    'E10': (-1832.0672, -1851.0872),
    'E11': (-1317.3674, -907.3084),
    'E12': (-1480.6018, -745.0728),
}
if not nodes:
    raise ValueError("The nodes dictionary is empty. Add the node coordinates in this cell.")

missing_exits = [e for e in EXITS if e not in nodes]
if missing_exits:
    raise ValueError(f"The following exits are missing from nodes: {', '.join(missing_exits)}")

print("\n📍 Available nodes:", " ".join(sorted(nodes.keys(), key=natural_key)))

# 3) Define start nodes, group sizes, candidate paths, and travel times

# 3-1) Number of start nodes
while True:
    try:
        num_starts = int(input("\nHow many start nodes will be evacuated? "))
        if num_starts <= 0:
            print("   ⚠️ The value must be at least 1.")
            continue
        break
    except:
        print("   ⚠️ Invalid number.")

start_nodes = []
people_at_start = {}
paths_by_start_exit = {}   # (start, exit) -> list of nodes
time_by_start_exit  = {}   # (start, exit) -> travel time in seconds

print("\n✅ Number of start nodes:", num_starts)

for i in range(1, num_starts+1):
    print("\n" + "="*60)
    print(f"🔹 Start node number {i}")

    # Start node
    while True:
        s = input("Start-node name (for example, N1): ").strip()
        if s not in nodes:
            print("   ⚠️ This node does not exist in nodes.")
            continue
        if s in start_nodes:
            print("   ⚠️ This node has already been entered as a start node.")
            continue
        start_nodes.append(s)
        break

    # Number of people at this node
    while True:
        try:
            cnt = float(input(f"Number of people at node {s}: "))
            if cnt < 0:
                print("   ⚠️ The number of people cannot be negative.")
                continue
            people_at_start[s] = cnt
            break
        except:
            print("   ⚠️ Invalid number.")

    # Candidate paths to E1, E6, and E7
    print(f"\n🧭 Now enter the candidate paths from this node ({s}) to each exit:")
    for ex in EXITS:
        while True:
            raw = input(f"Path nodes from {s} to {ex} (space- or comma-separated; first node = {s} and last node = {ex}): ").strip()
            toks = parse_nodes_line(raw)
            if not toks:
                print("   ⚠️ The path cannot be empty.")
                continue
            invalid = [t for t in toks if t not in nodes]
            if invalid:
                print(f"   ⚠️ Invalid nodes: {', '.join(invalid)}")
                continue
            if toks[0] != s:
                print(f"   ⚠️ The first path node must be {s}.")
                continue
            if toks[-1] != ex:
                print(f"   ⚠️ The last path node must be {ex}.")
                continue
            paths_by_start_exit[(s, ex)] = toks
            break

        while True:
            try:
                t_val = float(input(f"Travel time in seconds for path {s} → {ex} [{ ' → '.join(paths_by_start_exit[(s, ex)]) }]: "))
                time_by_start_exit[(s, ex)] = t_val
                break
            except:
                print("   ⚠️ Invalid travel-time value.")

# 4) Optional blocked nodes
blocked_raw = input("\nBlocked nodes (space- or comma-separated; leave blank if none): ").strip()
blocked_nodes = set(parse_nodes_line(blocked_raw)) if blocked_raw else set()

# 5) Upload exit images and run YOLO

images_by_exit = {}
for ex in EXITS:
    images_by_exit[ex] = upload_images_for_exit(ex, max_images=3)

print("\n⏳ Counting people at each exit with YOLOv8...")
auto_people = yolo_count_people_for_exits(images_by_exit, weights="yolov8n.pt", conf=0.25, iou=0.45, device="")
for ex in EXITS:
    print(f"   {ex}: Initial people detected from images ≈ {auto_people[ex]:.2f}")

# Dynamic person counts start from the YOLO estimates
people_dynamic = {ex: auto_people[ex] for ex in EXITS}

def get_rho(ex: str) -> float:
    """Return the current density at an exit: people / area."""
    area = exit_areas[ex]
    if area <= 0:
        return INF_COST
    return people_dynamic[ex] / area

# 6) Hazard centre and fear values for all nodes

danger = input("\nHazard-centre node: ").strip()
if danger not in nodes:
    raise ValueError("The hazard-centre node does not exist in nodes.")
danger_xy = nodes[danger]

fear_map = {}
rows_all = []
for n, xy in nodes.items():
    d = distance_m(xy, danger_xy)
    f = fear_func(d)
    fear_map[n] = f
    rows_all.append({"node": n, "distance_m": d, "fear": f})

with open(os.path.join(OUTPUT_DIR, "fear_all_nodes.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["node","distance_m","fear"])
    w.writeheader()
    w.writerows(sorted(rows_all, key=lambda r: natural_key(r["node"])))
print(f"✅ Saved: {os.path.join(OUTPUT_DIR, 'fear_all_nodes.csv')}")

# 7) Dynamic route evaluation
# Start nodes are processed from the largest group to the smallest group.
start_nodes_sorted = sorted(start_nodes, key=lambda s: people_at_start[s], reverse=True)
group_labels = {s: chr(65 + i) for i, s in enumerate(start_nodes_sorted)}

all_path_results = []
best_assignments = []

# Density history includes Stage 0 (before assigning any group).
density_history = [{"stage": 0, **{ex: get_rho(ex) for ex in EXITS}}]

print("\n" + "=" * 70)
print("Starting dynamic evacuation-route evaluation...")

for stage, s in enumerate(start_nodes_sorted, start=1):
    group = group_labels[s]
    print("\n" + "-" * 60)
    print(f"Group {group} | Start node: {s} | People: {people_at_start[s]:.0f}")

    candidate_results = []
    for ex_index, ex in enumerate(EXITS, start=1):
        nodes_in_path = paths_by_start_exit[(s, ex)]
        travel_time = time_by_start_exit[(s, ex)]
        avg_fear = sum(fear_map[n] for n in nodes_in_path) / len(nodes_in_path)
        exit_density = get_rho(ex)
        blocked = any(n in blocked_nodes for n in nodes_in_path)

        # Normalized components from Equation 9.
        time_term = WEIGHT_TIME * (travel_time / 60.0)
        density_term = WEIGHT_RHO * (exit_density / 7.0)
        fear_term = WEIGHT_FEAR * ((avg_fear - 3.93) / 4.07)
        final_cost = INF_COST if blocked else time_term + density_term + fear_term

        rec = {
            "route": f"{group}{ex_index}",
            "group": group,
            "start_node": s,
            "exit_node": ex,
            "nodes": " ".join(nodes_in_path),
            "num_nodes": len(nodes_in_path),
            "obstacle": "YES" if blocked else "NO",
            "travel_time_s": travel_time,
            "exit_density_person_m2": exit_density,
            "average_fear": avg_fear,
            "time_term": None if blocked else time_term,
            "density_term": None if blocked else density_term,
            "fear_term": None if blocked else fear_term,
            "final_cost": final_cost,
            "people_at_start": people_at_start[s],
            "stage": stage,
        }
        candidate_results.append(rec)
        all_path_results.append(rec)

    usable = [r for r in candidate_results if not r["obstacle"] == "YES"]
    if usable:
        best = min(usable, key=lambda r: r["final_cost"])
        best_assignments.append(best)
        chosen_exit = best["exit_node"]
        people_dynamic[chosen_exit] += people_at_start[s]
        print(
            f"Selected route: {best['route']} ({s} -> {chosen_exit}) | "
            f"Final cost = {best['final_cost']:.4f}"
        )
    else:
        print(f"All candidate routes for Group {group} are blocked. The group was not assigned.")

    density_history.append({"stage": stage, **{ex: get_rho(ex) for ex in EXITS}})

# 8) Create the route-evaluation table
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

route_table = pd.DataFrame(all_path_results)[[
    "route", "group", "obstacle", "travel_time_s",
    "exit_density_person_m2", "average_fear",
    "time_term", "density_term", "fear_term", "final_cost"
]].rename(columns={
    "route": "Route",
    "group": "Group",
    "obstacle": "Obstacle",
    "travel_time_s": "Time (s)",
    "exit_density_person_m2": "Exit density (Person/m²)",
    "average_fear": "Avg fear",
    "time_term": "Time term",
    "density_term": "Density term",
    "fear_term": "Fear term",
    "final_cost": "Final cost",
})

# Format the obstacle rows like the sample table.
formatted_table = route_table.copy()
for col in ["Time (s)", "Exit density (Person/m²)", "Avg fear"]:
    formatted_table[col] = formatted_table[col].map(lambda x: f"{x:.2f}")
for col in ["Time term", "Density term", "Fear term"]:
    formatted_table[col] = formatted_table[col].map(lambda x: "–" if pd.isna(x) else f"{x:.3f}")
formatted_table["Final cost"] = formatted_table["Final cost"].map(
    lambda x: "10¹⁰" if x >= INF_COST else f"{x:.2f}"
)

print("\nCost components and final evaluation of candidate evacuation routes")
display(formatted_table)

route_csv = os.path.join(OUTPUT_DIR, "evacuation_route_cost_table.csv")
route_table.to_csv(route_csv, index=False, encoding="utf-8-sig")
print(f"Saved: {route_csv}")

# 9) Plot a stacked comparison of cost components for all route groups
# Each group gets its own panel, similar to the requested sample figure.
# For blocked routes, the actual cost remains 10^10 in the table/CSV, while
# the purple bar is drawn at a visual cap so that the other routes stay readable.
plot_df = pd.DataFrame(all_path_results).copy()
groups_in_order = [group_labels[s] for s in start_nodes_sorted]

fig_width = max(10, 3.3 * len(groups_in_order))
fig, axes = plt.subplots(1, len(groups_in_order), figsize=(fig_width, 5.4), squeeze=False)
axes = axes.ravel()

component_labels = ["Time term", "Density term", "Fear term", "Obstacle penalty"]
component_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# A shared visual scale is based only on usable routes.
usable_costs = plot_df.loc[plot_df["final_cost"] < INF_COST, "final_cost"]
visual_top = max(0.75, float(usable_costs.max()) * 1.18 if not usable_costs.empty else 1.0)

for ax, group in zip(axes, groups_in_order):
    gdf = plot_df[plot_df["group"] == group].copy()
    x = np.arange(len(gdf))
    bar_width = 0.62

    time_vals = gdf["time_term"].fillna(0).to_numpy(float)
    density_vals = gdf["density_term"].fillna(0).to_numpy(float)
    fear_vals = gdf["fear_term"].fillna(0).to_numpy(float)
    blocked_mask = (gdf["obstacle"] == "YES").to_numpy()
    obstacle_vals = np.where(blocked_mask, visual_top * 0.96, 0.0)

    ax.bar(x, time_vals, width=bar_width, color=component_colors[0], edgecolor="white", linewidth=0.8)
    ax.bar(x, density_vals, width=bar_width, bottom=time_vals,
           color=component_colors[1], edgecolor="white", linewidth=0.8)
    ax.bar(x, fear_vals, width=bar_width, bottom=time_vals + density_vals,
           color=component_colors[2], edgecolor="white", linewidth=0.8)
    ax.bar(x, obstacle_vals, width=bar_width,
           color=component_colors[3], edgecolor="white", linewidth=0.8, hatch="//")

    for i, row in enumerate(gdf.itertuples(index=False)):
        if row.obstacle == "YES":
            ax.text(i, visual_top * 0.985, r"$10^{10}$", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        else:
            ax.text(i, row.final_cost + visual_top * 0.025, f"{row.final_cost:.2f}",
                    ha="center", va="bottom", fontsize=10)

    ax.set_title(f"Group {group}", fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(gdf["route"].tolist())
    ax.set_ylim(0, visual_top * 1.10)
    ax.set_ylabel("Final Cost" if ax is axes[0] else "")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Hide any unused panels, although normally none exist.
for ax in axes[len(groups_in_order):]:
    ax.set_visible(False)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=component_colors[0], label=component_labels[0]),
    Patch(facecolor=component_colors[1], label=component_labels[1]),
    Patch(facecolor=component_colors[2], label=component_labels[2]),
    Patch(facecolor=component_colors[3], hatch="//", label=component_labels[3]),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 0.96))
fig.suptitle("Stacked Cost Component Comparison for Candidate Route Groups",
             fontsize=14, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.88])

cost_component_plot = os.path.join(OUTPUT_DIR, "stacked_cost_component_comparison.png")
plt.savefig(cost_component_plot, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {cost_component_plot}")

# 10) Plot polished stage-by-stage exit-density changes
history_df = pd.DataFrame(density_history)

fig, ax = plt.subplots(figsize=(10.5, 5.8))
line_colors = {"E1": "#1f77b4", "E6": "#2ca02c", "E7": "#9467bd"}
markers = {"E1": "o", "E6": "s", "E7": "D"}

for ex in EXITS:
    ax.step(
        history_df["stage"],
        history_df[ex],
        where="post",
        marker=markers.get(ex, "o"),
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=2.8,
        color=line_colors.get(ex),
        label=rf"$\rho_{{{ex}}}$",
        zorder=3,
    )

    # Label the final value at the right-hand end of each line.
    final_x = history_df["stage"].iloc[-1]
    final_y = history_df[ex].iloc[-1]
    ax.annotate(
        rf"$\rho_{{{ex}}}$ = {final_y:.2f}",
        xy=(final_x, final_y),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        fontsize=11,
        color=line_colors.get(ex),
    )

ax.set_xlabel("Stage", fontsize=13)
ax.set_ylabel(r"Exit Density (person / m$^2$)", fontsize=13)
ax.set_title("Stage-by-Stage Exit Density Changes", fontsize=14, pad=12)
ax.set_xticks(history_df["stage"])
ax.grid(axis="y", linestyle="--", alpha=0.30)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="best")
ax.margins(x=0.08, y=0.12)
fig.tight_layout()

density_plot = os.path.join(OUTPUT_DIR, "exit_density_by_stage.png")
plt.savefig(density_plot, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {density_plot}")

# 11) Save density history and detailed route data
history_csv = os.path.join(OUTPUT_DIR, "exit_density_by_stage.csv")
history_df.to_csv(history_csv, index=False, encoding="utf-8-sig")

detailed_csv = os.path.join(OUTPUT_DIR, "evacuation_route_details.csv")
pd.DataFrame(all_path_results).to_csv(detailed_csv, index=False, encoding="utf-8-sig")

print(f"Saved: {history_csv}")
print(f"Saved: {detailed_csv}")

print("\nFinal exit conditions:")
for ex in EXITS:
    print(f"{ex}: people = {people_dynamic[ex]:.2f}, density = {get_rho(ex):.4f} person/m²")


# 12) Save a short scenario summary, create one ZIP file, and download it
summary_txt = os.path.join(OUTPUT_DIR, "scenario_summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(f"Scenario name: {scenario_name}\n")
    f.write(f"Safe scenario name: {safe_scenario_name}\n")
    f.write(f"Start nodes: {', '.join(start_nodes_sorted)}\n")
    f.write(f"Hazard-centre node: {danger}\n")
    f.write(f"Blocked nodes entered: {', '.join(sorted(blocked_nodes, key=natural_key)) if blocked_nodes else 'None'}\n")
    f.write("\nFinal exit conditions:\n")
    for ex in EXITS:
        f.write(
            f"{ex}: people = {people_dynamic[ex]:.2f}, "
            f"density = {get_rho(ex):.4f} person/m²\n"
        )

zip_base = f"/content/{safe_scenario_name}_results"
zip_path = shutil.make_archive(zip_base, "zip", OUTPUT_DIR)

print(f"\n✅ All outputs were collected in:")
print(f"   {zip_path}")

# Automatically download the ZIP file in Google Colab.
files.download(zip_path)
