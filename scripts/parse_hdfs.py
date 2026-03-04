import sys
import os
import re
import pandas as pd
from datetime import datetime

# Add the project root to sys.path to import Drain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drain import Drain

def parse_hdfs_raw(log_path, label_path, output_path, max_lines=2000):
    print(f"Starting HDFS log parsing (max_lines={max_lines})...")
    
    # Standard regex for HDFS cleaning
    rex = [
        r'blk_-?\d+', # block IDs
        r'(\d+\.){3}\d+', # IPs
        r'\d+' # numbers
    ]
    
    parser = Drain(depth=4, st=0.5, rex=rex)
    
    # Regex for field extraction from HDFS logs
    # Format: 081109 203615 148 INFO dfs.DataNode$PacketResponder: ...
    log_pattern = re.compile(r'^(\d+) (\d+) (\d+) (INFO|WARN|ERROR|FATAL) ([\w.$]+): (.*)$')
    
    parsed_data = []
    
    with open(log_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            
            line = line.strip()
            match = log_pattern.match(line)
            
            if match:
                date, time, pid, level, component, content = match.groups()
                
                # Extract BlockID for mapping
                block_match = re.search(r'blk_-?\d+', content)
                block_id = block_match.group() if block_match else "None"
                
                event_id, template = parser.add_log_line(content)
                
                parsed_data.append({
                    'LineId': i + 1,
                    'Date': date,
                    'Time': time,
                    'Pid': pid,
                    'Level': level,
                    'Component': component,
                    'Content': content,
                    'BlockId': block_id,
                    'EventId': event_id,
                    'EventTemplate': template
                })
            else:
                print(f"Warning: Line {i+1} did not match pattern: {line[:50]}...")

    df = pd.DataFrame(parsed_data)
    
    # Load labels
    print(f"Loading labels from {label_path}...")
    labels_df = pd.read_csv(label_path)
    
    # Merge labels
    print("Merging labels with parsed data...")
    # Clean BlockId in labels if necessary (ensure they match the extracted format)
    df = df.merge(labels_df, on='BlockId', how='left')
    
    # Handle missing labels
    df['Label'] = df['Label'].fillna('Unknown')
    
    # Save output
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} lines to {output_path}")

if __name__ == "__main__":
    LOG_PATH = r"c:\secure-log\data\HDFS\HDFS.log"
    LABEL_PATH = r"c:\secure-log\data\HDFS\anomaly_label.csv"
    OUTPUT_PATH = r"c:\secure-log\data\HDFS\2k-parsed-structured.csv"
    
    parse_hdfs_raw(LOG_PATH, LABEL_PATH, OUTPUT_PATH, max_lines=2000)
