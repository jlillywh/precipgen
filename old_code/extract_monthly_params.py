import pandas as pd
import sys

# Read the results file
results_file = 'results_australian.xlsx'
print('Reading results from:', results_file)
print()

# Read the Parameters_Output sheet
df = pd.read_excel(results_file, sheet_name='Parameters_Output')

# The monthly parameters start at row 2 (index 1) and go for 12 months
# Find where the monthly parameters section starts
monthly_start = None
for i, row in df.iterrows():
    if str(row.iloc[0]) == 'Month':
        monthly_start = i
        break

if monthly_start is not None:
    # Extract the monthly parameters (next 12 rows after the header)
    monthly_data = df.iloc[monthly_start:monthly_start+13].copy()  # Header + 12 months
    
    # Clean up the data - set proper column names
    monthly_data.columns = monthly_data.iloc[0]  # Use first row as column names
    monthly_data = monthly_data.iloc[1:]  # Remove the header row
    monthly_data = monthly_data.reset_index(drop=True)
    
    print("Monthly Parameters:")
    print(monthly_data)
    print()
    
    # Extract January results specifically
    january_data = monthly_data[monthly_data['Month'] == 1]
    if not january_data.empty:
        print("JANUARY RESULTS:")
        jan_row = january_data.iloc[0]
        print(f"Month: {jan_row['Month']}")
        print(f"P(W|W): {jan_row['p_ww']:.6f}")
        print(f"P(W|D): {jan_row['p_wd']:.6f}")
        print(f"Alpha:  {jan_row['alpha']:.6f}")
        print(f"Beta:   {jan_row['beta']:.6f}")
        print()
        
        # Compare with GoldSim Australian results
        print("COMPARISON WITH GOLDSIM (Australian Station):")
        print("GoldSim Results:")
        print("P(W|W): 0.2923588")
        print("P(W|D): 0.06011854")
        print("Alpha:  0.7124466")
        print("Beta:   9.424629")
        print()
        print("Python Results:")
        print(f"P(W|W): {jan_row['p_ww']:.6f}")
        print(f"P(W|D): {jan_row['p_wd']:.6f}")
        print(f"Alpha:  {jan_row['alpha']:.6f}")
        print(f"Beta:   {jan_row['beta']:.6f}")
        print()
        print("Differences:")
        print(f"P(W|W): {0.2923588 - jan_row['p_ww']:.6f}")
        print(f"P(W|D): {0.06011854 - jan_row['p_wd']:.6f}")
        print(f"Alpha:  {0.7124466 - jan_row['alpha']:.6f}")
        print(f"Beta:   {9.424629 - jan_row['beta']:.6f}")
        
        # Show all months comparison
        print("\n" + "="*80)
        print("FULL MONTHLY COMPARISON:")
        print("="*80)
        
        # GoldSim results for all months (Australian station)
        goldsim_results = {
            1: {'pww': 0.2923588, 'pwd': 0.06011854, 'alpha': 0.7124466, 'beta': 9.424629},
            2: {'pww': 0.3515358, 'pwd': 0.06014335, 'alpha': 0.7574874, 'beta': 9.274518},
            3: {'pww': 0.32493, 'pwd': 0.0671064, 'alpha': 0.6318583, 'beta': 8.386636},
            4: {'pww': 0.4491682, 'pwd': 0.09971689, 'alpha': 0.778419, 'beta': 5.326887},
            5: {'pww': 0.4750567, 'pwd': 0.1603646, 'alpha': 0.815061, 'beta': 4.617329},
            6: {'pww': 0.5203704, 'pwd': 0.1984849, 'alpha': 0.8309303, 'beta': 4.205232},
            7: {'pww': 0.511648, 'pwd': 0.2085661, 'alpha': 0.9125425, 'beta': 3.415679},
            8: {'pww': 0.5459057, 'pwd': 0.2098672, 'alpha': 0.9480786, 'beta': 3.510221},
            9: {'pww': 0.446186, 'pwd': 0.1856678, 'alpha': 0.8599602, 'beta': 4.827606},
            10: {'pww': 0.4104046, 'pwd': 0.1275381, 'alpha': 0.8051432, 'beta': 5.757591},
            11: {'pww': 0.4078212, 'pwd': 0.09927741, 'alpha': 0.7581188, 'beta': 6.91184},
            12: {'pww': 0.356044, 'pwd': 0.08471075, 'alpha': 0.7125986, 'beta': 7.951274}
        }
        
        print(f"{'Month':<5} {'Parameter':<8} {'GoldSim':<10} {'Python':<10} {'Difference':<12}")
        print("-" * 50)
        
        for month in range(1, 13):
            if month <= len(monthly_data):
                python_row = monthly_data.iloc[month-1]
                gs = goldsim_results[month]
                
                print(f"{month:<5} {'P(W|W)':<8} {gs['pww']:<10.6f} {python_row['p_ww']:<10.6f} {gs['pww'] - python_row['p_ww']:<12.6f}")
                print(f"{'':<5} {'P(W|D)':<8} {gs['pwd']:<10.6f} {python_row['p_wd']:<10.6f} {gs['pwd'] - python_row['p_wd']:<12.6f}")
                print(f"{'':<5} {'Alpha':<8} {gs['alpha']:<10.6f} {python_row['alpha']:<10.6f} {gs['alpha'] - python_row['alpha']:<12.6f}")
                print(f"{'':<5} {'Beta':<8} {gs['beta']:<10.6f} {python_row['beta']:<10.6f} {gs['beta'] - python_row['beta']:<12.6f}")
                print()
else:
    print("Could not find monthly parameters section in the results file")
    print("Available data:")
    print(df.head(20))