# 📊 CSV Data Files

## Files in This Repository

### ✅ Included (Under 50MB)

**Tax_Administration_s_Real_Estate_-_Assessed_Values.csv** (~28 MB)
- **Status**: ✅ Included in repository
- **Purpose**: Core data file for property value forecasting
- **Contains**: Property assessed values by year (TAXYR, PARID, APRTOT, etc.)
- **Used by**: All forecasting functions in `sts/data/fairfax_loader.py`

This is the **only required file** for the application to work.

---

## Files NOT Included (Over 50MB or Not Used)

The following files exceed GitHub's 50MB recommendation or are not currently used by the application:

| File | Size | Status | Reason |
|------|------|--------|--------|
| `Tax_Administration_s_Real_Estate_-_Parcel_Data.csv` | 56 MB | ❌ Not included | Over 50MB limit, not currently used |
| `Tax_Administration_s_Real_Estate_-_Sales_Data.csv` | 114 MB | ❌ Not included | Over 100MB limit, not currently used |
| `Tax_Administration_s_Real_Estate_-_Dwelling_Data.csv` | 72 MB | ❌ Not included | Over 50MB limit, not currently used |
| `Tax_Administration_s_Real_Estate_-_Land_Data.csv` | 19 MB | ❌ Not included | Not currently used |
| `Tax_Administration_s_Real_Estate_-_Commercial_Data.csv` | 1.5 MB | ❌ Not included | Not currently used |

These files are excluded via `.gitignore` to keep the repository size manageable.

---

## What You Need to Do

### If You Cloned This Repository

**Nothing!** The required CSV file (`Tax_Administration_s_Real_Estate_-_Assessed_Values.csv`) is already included.

Just run:
```bash
pip3 install -r requirements.txt
pip3 install prophet>=1.1.5
pip3 install -e .
streamlit run apps/app.py
```

### If You Want Additional Data (Optional)

If you want to enhance the application with additional data files in the future, download them from:

**Fairfax County Open Data Portal**: https://data.fairfaxcounty.gov/

Search for: **"Tax Administration Real Estate"**

Download and place in this `csvs/` directory.

---

## Why Only One File?

### Code Analysis

Looking at `sts/data/fairfax_loader.py`, the application only uses:

1. **Tax_Administration_s_Real_Estate_-_Assessed_Values.csv** (Required)
   - Used in: `load_fairfax_assessed_values()`
   - Used in: `get_available_districts()`
   - Used in: `get_district_summary()`
   - Contains all data needed for forecasting

2. **Tax_Administration_s_Real_Estate_-_Parcel_Data.csv** (Referenced but not required)
   - Has a parameter in `load_fairfax_assessed_values()` but not currently used
   - Can be added later if needed for additional metadata

### File Size Constraints

- **GitHub limit**: 50 MB recommended, 100 MB hard limit
- **Assessed Values**: 28 MB ✅ (fits comfortably)
- **Parcel Data**: 56 MB ❌ (exceeds recommendation)
- **Sales Data**: 114 MB ❌ (exceeds hard limit)

---

## Data Structure

### Tax_Administration_s_Real_Estate_-_Assessed_Values.csv

**Key Columns Used:**
- `PARID` - Parcel ID (first 4 digits = district code)
- `TAXYR` - Tax year
- `APRTOT` - Total appraised value
- `PRITOT` - Total prior year value

**Sample Data:**
```csv
OBJECTID,PARID,TAXYR,APRLAND,APRBLDG,APRTOT,PRILAND,PRIBLDG,PRITOT,FLAG4_DESC
1,0804 05190022,2025,290000,273820,563820,280000,262990,542990,No Exemption
2,0601 31  0001,2025,367000,432160,799160,337000,398250,735250,No Exemption
```

**How It's Used:**
1. Extract district from `PARID` (first 4 digits)
2. Group by `TAXYR` and district
3. Aggregate `APRTOT` (total appraised value)
4. Create time series for forecasting

---

## Future Enhancements (Phase 2+)

If you want to add more data sources:

### Option 1: Sales Data (for validation)
- Download `Tax_Administration_s_Real_Estate_-_Sales_Data.csv`
- Use to validate forecasts against actual sales
- Note: 114 MB - would need Git LFS or external storage

### Option 2: Parcel Metadata
- Download `Tax_Administration_s_Real_Estate_-_Parcel_Data.csv`
- Add property characteristics (zoning, location, etc.)
- Note: 56 MB - would need Git LFS or external storage

### Option 3: External Regressors
- CPI/Inflation data
- Interest rates
- Employment data
- See `sts/data/external_regressors.py` for framework

---

## Troubleshooting

### Issue: "CSV file not found"

**If you see this error:**
```
FileNotFoundError: csvs/Tax_Administration_s_Real_Estate_-_Assessed_Values.csv
```

**Solution:**
The file should be in the repository. If it's missing:
1. Check you cloned the full repository
2. Verify the file exists: `ls -lh csvs/*.csv`
3. If missing, download from: https://data.fairfaxcounty.gov/

### Issue: "Want to use additional CSV files"

**If you want Parcel Data, Sales Data, etc.:**

1. Download from Fairfax County Open Data Portal
2. Place in `csvs/` directory
3. Update code in `sts/data/fairfax_loader.py` to use them
4. Note: These files are gitignored, so they won't be committed

---

## Summary

✅ **What's Included**: `Tax_Administration_s_Real_Estate_-_Assessed_Values.csv` (28 MB)  
❌ **What's Not**: Other CSV files (too large or not used)  
🎯 **What You Need**: Just clone and run - everything is ready!

---

**The application is fully functional with just the included CSV file!** 🚀

*Last Updated: January 14, 2026*
