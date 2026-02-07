import os
import glob
import argparse
import torch
import torch.multiprocessing as mp
import time
from queue import Empty

from process_lma_features import process_single_video

# Import the Model Class (Same as in your original script)
from moge.model.v2 import MoGeModel

def worker_process(gpu_id, queue, output_dir, viz):
    """
    1. Initializes models on the assigned GPU (Run once per worker)
    2. Consumes files from the queue until empty
    """
    process_name = mp.current_process().name
    device = f"cuda:{gpu_id}"
    print(f"[{process_name}] Launching on {device}...")

    # --- A. LOAD MODELS ONCE ---
    try:
        # Load NLF
        nlf_model = torch.jit.load('models/nlf_l_multi_0.3.2.torchscript').to(device).eval()
        # Load MoGe
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        print(f"[{process_name}] Models loaded. Ready to work.")
    except Exception as e:
        print(f"[{process_name}] CRITICAL MODEL LOAD FAIL: {e}")
        return

    # --- B. PROCESSING LOOP ---
    processed_count = 0
    while True:
        try:
            # Get next file from queue (timeout allows clean exit if queue stuck)
            video_path = queue.get(timeout=5)
        except Empty:
            # Queue is empty, work is done
            break

        try:
            # CALL YOUR FUNCTION
            process_single_video(
                video_path=video_path, 
                output_dir=output_dir, 
                nlf_model=nlf_model, 
                moge_model=moge_model, 
                device=device, 
                viz=viz
            )
            processed_count += 1
        except Exception as e:
            print(f"[{process_name}] FAILED on {os.path.basename(video_path)}: {e}")
        finally:
            # Signal that this specific task is complete
            queue.task_done()

    print(f"[{process_name}] Done. Processed {processed_count} videos.")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Master Runner")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .mp4 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--workers_per_gpu", type=int, default=6, 
                        help="Number of parallel processes per GPU. (Default: 6)")
    parser.add_argument("--viz", action="store_true", help="Enable debug video generation")
    args = parser.parse_args()

    # 1. Setup
    mp.set_start_method('spawn', force=True) # Critical for CUDA
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Collect Videos
    print(f"Scanning {args.input_dir}...")
    video_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    total_files = len(video_files)
    print(f"Found {total_files} videos.")

    if total_files == 0:
        return

    # 3. Fill Queue
    task_queue = mp.JoinableQueue()
    for v in video_files:
        task_queue.put(v)

    # 4. Launch Workers
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")
    if num_gpus == 0:
        print("No GPUs found! Exiting.")
        return

    workers = []
    print(f"Spawning {args.workers_per_gpu} workers per GPU ({num_gpus * args.workers_per_gpu} total)...")
    
    for gpu_id in range(num_gpus):
        for _ in range(args.workers_per_gpu):
            p = mp.Process(
                target=worker_process, 
                args=(gpu_id, task_queue, args.output_dir, args.viz)
            )
            p.start()
            workers.append(p)

    # 5. Monitor
    print("\n--- PROCESSING STARTED ---")
    start_time = time.time()
    
    try:
        # Wait for queue to be empty
        while not task_queue.empty():
            remaining = task_queue.qsize()
            done = total_files - remaining
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            
            print(f"Progress: {done}/{total_files} | Rate: {rate:.2f} vids/sec", end='\r')
            time.sleep(5)
            
        # Final join to ensure all items are actually processed
        task_queue.join()
        
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Terminating workers...")
        for p in workers:
            p.terminate()
            
    print(f"\n\nDone! Processed {total_files} videos in {(time.time() - start_time)/60:.1f} minutes.")

if __name__ == "__main__":
    main()
