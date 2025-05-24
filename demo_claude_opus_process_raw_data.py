import pandas as pd
import os
from io import StringIO

# Create sample raw keystroke data
sample_raw_data = """P,h,1000
R,h,1150
P,e,1200
R,e,1350
P,l,1400
R,l,1500
P,l,1550
R,l,1650
P,o,1700
R,o,1850
P,w,1900
P,o,1950
R,w,2000
R,o,2100
P,r,2150
P,l,2200
R,l,2300
P,d,2350
R,d,2500"""

# Create another sample with errors
sample_with_errors = """P,t,1000
R,t,1100
P,e,1150
P,s,1300
R,s,1400
R,t,1500
P,i,1600
R,i,1700
P,n,1750
P,g,1900
R,n,1950
R,g,2050"""

def demonstrate_extraction():
    """Demonstrate the TypeNet feature extraction process"""
    
    # Create temporary files
    os.makedirs('demo_data/user_1001', exist_ok=True)
    os.makedirs('demo_data/user_1002', exist_ok=True)
    
    # Write sample data
    with open('demo_data/user_1001/1_1_1_1001.csv', 'w') as f:
        f.write(sample_raw_data)
    
    with open('demo_data/user_1002/1_2_1_1002.csv', 'w') as f:
        f.write(sample_with_errors)
    
    # Import the extractor (assuming it's in the same directory or in Python path)
    from typenet_extraction import TypeNetFeatureExtractor
    
    # Create extractor
    extractor = TypeNetFeatureExtractor()
    
    print("=== TypeNet Feature Extraction Demo ===\n")
    
    # Process the demo dataset
    print("Processing demo dataset...")
    features_df = extractor.process_dataset('demo_data', 'demo_features.csv')
    
    print("\n=== Extracted Features ===")
    print(features_df.to_string(index=False))
    
    print("\n=== Feature Statistics ===")
    print(f"Total keypair features: {len(features_df)}")
    print(f"Valid features: {features_df['valid'].sum()}")
    print(f"Invalid features: {(~features_df['valid']).sum()}")
    
    print("\n=== Timing Feature Ranges (valid records only) ===")
    valid_df = features_df[features_df['valid']]
    if not valid_df.empty:
        for feature in ['HL', 'IL', 'PL', 'RL']:
            if feature in valid_df.columns and valid_df[feature].notna().any():
                print(f"{feature}: min={valid_df[feature].min():.0f}ms, "
                      f"max={valid_df[feature].max():.0f}ms, "
                      f"mean={valid_df[feature].mean():.0f}ms")
    
    print("\n=== Error Analysis ===")
    error_df = features_df[~features_df['valid']]
    if not error_df.empty:
        print("Errors found:")
        for error_type, count in error_df['error_description'].value_counts().items():
            print(f"  - {error_type}: {count} occurrences")
        
        print("\nExample error records:")
        print(error_df[['key1', 'key2', 'error_description']].head(5).to_string(index=False))
    else:
        print("No errors found in the dataset!")
    
    # Cleanup
    import shutil
    shutil.rmtree('demo_data')
    os.remove('demo_features.csv')
    
    return features_df

def show_algorithm_explanation():
    """Explain the parentheses matching algorithm"""
    
    print("=== Parentheses Matching Algorithm Explanation ===\n")
    
    print("The algorithm works like matching parentheses:")
    print("1. Each 'P' (press) is like an opening parenthesis '('")
    print("2. Each 'R' (release) is like a closing parenthesis ')'")
    print("3. We maintain a stack for each unique key")
    print("4. When we see a press, we push it onto the stack")
    print("5. When we see a release, we pop from the stack to match it")
    print("\nExample sequence: P(h) R(h) P(e) P(l) R(e) R(l)")
    print("Matches: (h matched), (e matched with overlap), (l matched)")
    
    print("\n=== Error Types Handled ===")
    print("1. Missing release: P(x) but no R(x) - key held down indefinitely")
    print("2. Orphan release: R(x) without preceding P(x) - release without press")
    print("3. Invalid timing: Negative hold time or press latency")
    print("4. Propagated errors: If key1 or key2 is invalid, the pair is invalid")

if __name__ == "__main__":
    # Show algorithm explanation
    show_algorithm_explanation()
    
    print("\n" + "="*50 + "\n")
    
    # Run the demonstration
    demo_df = demonstrate_extraction()


