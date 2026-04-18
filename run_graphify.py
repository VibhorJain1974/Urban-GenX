#!/usr/bin/env python3
"""Graphify pipeline runner for Urban-GenX"""

import json
import sys
from pathlib import Path

def main():
    # Add graphify-out to path
    graphify_out = Path('graphify-out')
    graphify_out.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("GRAPHIFY PIPELINE: Urban-GenX")
    print("=" * 60)
    
    # Step 2: Detect files
    print("\n[Step 2] Detecting files...")
    try:
        from graphify.detect import detect
        result = detect(Path('.'))
        
        detect_file = graphify_out / '.graphify_detect.json'
        detect_file.write_text(json.dumps(result, indent=2))
        
        total_files = result.get('total_files', 0)
        total_words = result.get('total_words', 0)
        
        print(f"\nCorpus: {total_files} files · ~{total_words:,} words")
        
        files_by_type = result.get('files', {})
        for ftype, files in files_by_type.items():
            if files:
                print(f"  {ftype:10} {len(files):3} files")
        
        if total_files == 0:
            print("\n❌ No supported files found.")
            return 1
        
        if total_words > 2_000_000 or total_files > 200:
            print(f"\n⚠️  Large corpus detected: {total_words:,} words, {total_files} files")
            print("   Top directories by file count:")
            # Get subdirs with most files
            subdir_counts = {}
            for ftype, files in files_by_type.items():
                for f in files:
                    p = Path(f)
                    if len(p.parts) > 1:
                        top = p.parts[0]
                        subdir_counts[top] = subdir_counts.get(top, 0) + 1
            
            for subdir, count in sorted(subdir_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"     - {subdir:30} {count:3} files")
            
            print("\n   Recommendation: run on a specific subdirectory to speed up extraction.")
            print("   Example: /graphify src/")
            # For now, continue with full corpus
        
        # Check for video files
        video_files = files_by_type.get('video', [])
        if video_files:
            print(f"\n[Step 2.5] Found {len(video_files)} video/audio file(s) - would need transcription")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
