import pandas as pd
import numpy as np
import os
import argparse

def preprocess_artemis_data(spacecraft_file, moon_file, output_dir):
    """
    Preprocess Artemis mission data for the orbit prediction model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse data files with more robust settings
    print(f"Parsing spacecraft data from {spacecraft_file}...")
    sc_data = pd.read_csv(spacecraft_file)
    
    print(f"Parsing moon data from {moon_file}...")
    moon_data = pd.read_csv(moon_file)
    
    print("Spacecraft columns:", sc_data.columns.tolist())
    
    # Handle column selection more robustly
    sc_columns = {
        'time': sc_data.columns[0],  # First column should be time
        's_x': 'X',
        's_y': 'Y', 
        's_z': 'Z',
        'v_x': 'VX',
        'v_y': 'VY',
        'v_z': 'VZ'
    }
    
    moon_columns = {
        'time': moon_data.columns[0],  # First column should be time
        'm_x': 'X',
        'm_y': 'Y',
        'm_z': 'Z'
    }
    
    # Create renamed dataframes
    sc_renamed = sc_data.rename(columns={
        sc_columns['time']: 'time',
        sc_columns['s_x']: 's_x',
        sc_columns['s_y']: 's_y',
        sc_columns['s_z']: 's_z',
        sc_columns['v_x']: 'v_x',
        sc_columns['v_y']: 'v_y', 
        sc_columns['v_z']: 'v_z'
    })
    
    moon_renamed = moon_data.rename(columns={
        moon_columns['time']: 'time',
        moon_columns['m_x']: 'm_x',
        moon_columns['m_y']: 'm_y',
        moon_columns['m_z']: 'm_z'
    })
    
    # Select only needed columns
    sc_renamed = sc_renamed[['time', 's_x', 's_y', 's_z', 'v_x', 'v_y', 'v_z']]
    moon_renamed = moon_renamed[['time', 'm_x', 'm_y', 'm_z']]
    
    # Merge datasets on time
    print("Merging spacecraft and moon data...")
    merged_data = pd.merge(sc_renamed, moon_renamed, on='time', how='inner')
    
    # Convert time to minutes from start
    start_time = merged_data['time'].min()
    merged_data['t'] = (merged_data['time'] - start_time) * 1440
    
    print(f"Merged data contains {len(merged_data)} rows")
    
    # Calculate distances 
    merged_data['earth_dist'] = np.sqrt(
        merged_data['s_x']**2 + merged_data['s_y']**2 + merged_data['s_z']**2
    )
    
    merged_data['moon_dist'] = np.sqrt(
        (merged_data['s_x'] - merged_data['m_x'])**2 + 
        (merged_data['s_y'] - merged_data['m_y'])**2 + 
        (merged_data['s_z'] - merged_data['m_z'])**2
    )
    
    # Detect mission phase
    conditions = [
        merged_data['earth_dist'] < 50000,  # Near Earth
        merged_data['moon_dist'] < 50000    # Near Moon
    ]
    choices = [0, 2]  # 0=Earth, 1=Transit (default), 2=Moon
    merged_data['phase'] = np.select(conditions, choices, default=1)
    
    # Create final dataset
    final_data = merged_data[['t', 's_x', 's_y', 's_z', 'v_x', 'v_y', 'v_z', 
                             'm_x', 'm_y', 'm_z', 'phase']]
    
    # Save processed data
    output_file = os.path.join(output_dir, 'processed_data.csv')
    final_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    # Create simple visualization
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.scatter(final_data['s_x'], final_data['s_y'], final_data['s_z'], 
                  c=final_data['phase'], s=2)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Artemis I Trajectory')
        
        plt.savefig(os.path.join(output_dir, 'trajectory.png'))
        print(f"Visualization saved to {os.path.join(output_dir, 'trajectory.png')}")
    except ImportError:
        print("Matplotlib not available, skipping visualization")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Artemis mission data')
    parser.add_argument('--spacecraft', required=True, help='Path to spacecraft data')
    parser.add_argument('--moon', required=True, help='Path to moon data')
    parser.add_argument('--output', default='change_processed_data', help='Output directory')
    
    args = parser.parse_args()
    preprocess_artemis_data(args.spacecraft, args.moon, args.output)