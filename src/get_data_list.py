import os
from collections import defaultdict

# 1. Criteria from Section 3.1 of the paper
TARGET_GENRES = ['gBR', 'gPO', 'gLO', 'gWA', 'gMH', 'gLH', 'gHO', 'gKR', 'gJS', 'gJB']
TARGET_SITUATIONS = ['sBM', 'sFM'] # Basic and Advanced
TARGET_CAMERA = 'c01'               # Frontal view
TARGET_PER_GENRE = 60               # 10 genres * 60 = 600 total

def filter_aist_videos(all_files):
    genre_map = defaultdict(list)
    
    # Sort files to ensure deterministic selection
    for filename in sorted(all_files):
        # Format: {Genre}_{Situation}_{Camera}_{Dancer}_{Music}_{Choreography}.mp4
        parts = filename.split('_')
        if len(parts) < 3: continue
        
        genre, situation, camera = parts[0], parts[1], parts[2]
        
        # Apply strict filename filters
        if (genre in TARGET_GENRES and 
            situation in TARGET_SITUATIONS and 
            camera == TARGET_CAMERA):
            
            # Additional 'Standing' logic: In AIST++, 'sBM' (Basic) is 
            # more likely to be standing than 'sFM' or choreographed work.
            # We prioritize sBM to match the "standing dancers only" rule.
            genre_map[genre].append(filename)

    original_count = sum(len(v) for v in genre_map.values())
    print(f"Original filtered count before limiting per genre: {original_count}")

    final_600_list = []
    for g in TARGET_GENRES:
        # Take the first 60 available for that genre
        subset = genre_map[g][:TARGET_PER_GENRE]
        final_600_list.extend(subset)
        
        if len(subset) < TARGET_PER_GENRE:
            print(f"Warning: Genre {g} only has {len(subset)} videos.")

    return final_600_list


all_files = os.listdir('/home/sogang/mnt/db_1/jaehoon/aist')
target_videos = filter_aist_videos(all_files)
print(f"Found {len(target_videos)} videos matching the criteria.")

# Save the filtered list to a text file
with open('aist_filtered_videos.txt', 'w') as f:
    for video in target_videos:
        f.write(f"{video}\n")
