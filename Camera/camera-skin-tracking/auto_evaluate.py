import os
import glob
import sys
import argparse
import csv
from contextlib import redirect_stdout, redirect_stderr

import evaluate_tracking

def find_video(folder_path, subfolder_name):
    """Finds video file inside a specific subfolder (e.g., folder/video_cam1/file.mp4)"""
    search_pattern = os.path.join(folder_path, subfolder_name, "*")
    matches = glob.glob(search_pattern)
    video_exts = ['.mp4', '.avi', '.webm', '.mkv']
    valid_matches = [m for m in matches if any(m.lower().endswith(ext) for ext in video_exts)]
    
    if valid_matches:
         return valid_matches[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="Automated tracking evaluation for all recordings.")
    parser.add_argument("--axes", type=str, default="x,y,z,roll,pitch,yaw", help="Axes to evaluate (e.g. 'x,y', see evaluate_tracking.py for details).")
    parser.add_argument("--measurements_dir", type=str, default=os.path.normpath(os.path.join(os.getcwd(), "..", "measurements_with_imu")), help="Path to the main measurements directory.")
    parser.add_argument("--aruco_camera_offset", type=str, default="0,0,0", help="Optional offset vector between camera and ArUco for manual measurements (x,y,z mm)")
    args = parser.parse_args()

    measurements_dir = args.measurements_dir
    
    if not os.path.exists(measurements_dir):
        print(f"Measurements directory does not exist: {measurements_dir}")
        return

    # List to store evaluation results
    results_list = []

    print(f"Starting automated evaluation in directory: {measurements_dir}\n")

    # Save original arguments
    original_argv = sys.argv.copy()

    scores_dir = "scores"
    os.makedirs(scores_dir, exist_ok=True)
    output_file = os.path.join(scores_dir, "auto_evaluate_results.csv")
    
    # Create a new empty CSV file and write headers to log live results
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Measurement", "MAE", "RMSE"])
        writer.writeheader()

    for item in os.listdir(measurements_dir):
        folder_path = os.path.join(measurements_dir, item)
        
        if os.path.isdir(folder_path):
            print(f"[{item}] Reading videos...")
            
            vid_probe = find_video(folder_path, "video_cam1")  # from probe
            vid_ext = find_video(folder_path, "video_cam2")    # from external (reference)
            
            if not vid_probe or not vid_ext:
                print(f"  -> Skipping. Missing required video files (video_cam1 and video_cam2) in {item}")
                res = {"Measurement": item, "MAE": "Missing files", "RMSE": "Missing files"}
                results_list.append(res)
                # Live write
                with open(output_file, mode='a', newline='', encoding='utf-8') as f:
                    csv.DictWriter(f, fieldnames=["Measurement", "MAE", "RMSE"]).writerow(res)
                continue
                
            print(f"  -> Calculating RMSE error...")
            
            axes_to_use = args.axes
            if "__" in item:
                parts = item.split("__", 1)
                axes_to_use = parts[1]
                print(f"     [INFO] Detected axis override from folder name: {axes_to_use}")
            
            # Set arguments for evaluate_tracking using the folder name as prefix
            gt_path = os.path.join(folder_path, "video_cam2", "ground_truth.csv")
            of_path = os.path.join(folder_path, "video_cam1", "optical_flow.csv")
            sys.argv = [
                "evaluate_tracking.py",
                "--video_probe", vid_probe,
                "--video_ext", vid_ext,
                "--output_prefix", item,
                "--headless",
                "--save_gt", gt_path,
                "--save_of", of_path,
                "--axes", axes_to_use,
                "--aruco_camera_offset", args.aruco_camera_offset
            ]
            
            result_tuple = None
            res = {}
            try:
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull), redirect_stderr(fnull):
                       result_tuple = evaluate_tracking.main()
                
                if result_tuple is not None:
                    mae_val, rmse_val = result_tuple
                    print(f"  -> Completed! Results | MAE: {mae_val:.3f} | RMSE: {rmse_val:.3f}")
                    res = {"Measurement": item, "MAE": round(mae_val, 3), "RMSE": round(rmse_val, 3)}
                else:
                    print(f"  -> No valid result returned (got None)")
                    res = {"Measurement": item, "MAE": "Error", "RMSE": "Error"}
            except Exception as e:
                print(f"  -> Exception occurred during evaluation: {e}")
                res = {"Measurement": item, "MAE": "Exception", "RMSE": "Exception"}
            
            results_list.append(res)
            # Live write
            with open(output_file, mode='a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=["Measurement", "MAE", "RMSE"]).writerow(res)

    # Restore original sys.argv command line arguments
    sys.argv = original_argv

    # Print summary table
    if not results_list:
        print("\nNo results gathered. Please check the measurements directory structure.")
        return
        
    print("\n" + "="*45)
    print("TRACKING EVALUATION SUMMARY")
    print("="*45)
    print(f"{'Measurement':<30} | {'MAE':<10} | {'RMSE'}")
    print("-" * 60)
    for res in results_list:
        print(f"{res['Measurement']:<30} | {res['MAE']:<10} | {res['RMSE']}")
    print("="*60)
    
    print(f"\nResults were saved progressively to: {output_file}")

if __name__ == "__main__":
    main()
