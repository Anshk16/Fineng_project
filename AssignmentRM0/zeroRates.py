import numpy as np
import pandas as pd
from datetime import date, datetime
from yearfrac import yearfrac, mod

# Se hai già una funzione yearfrac.py e un enum mod, importali:
# from yearfrac import yearfrac, mod
# In MATLAB la convenzione 3 era 30/360 ISDA, 
# che in un'ipotetica enum Python potresti chiamare mod.EU_30_360
# Altrimenti, adatta in base alla tua implementazione.

def to_date(x):
    """
    Converte un oggetto in datetime.date se è un pd.Timestamp o datetime.datetime.
    Se l'oggetto è già un datetime.date, lo restituisce così com'è.
    """
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    return x

def zeroRates(dates_df: pd.DataFrame, 
              discounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i tassi zero-coupon (in percentuale) a partire da un riferimento fisso 
    (datetime(2023, 2, 2) in MATLAB) e dalle colonne 'Date' e 'Discount Factor' 
    di due DataFrame di pandas.

    Parametri:
    - dates_df: DataFrame con almeno una colonna 'Date'.
    - discounts_df: DataFrame con almeno una colonna 'Discount Factor'.
      Deve avere la stessa lunghezza di dates_df.

    Restituisce:
    - Un DataFrame con due colonne: 'Date' e 'Zero Rate' (in percentuale).
    """

    # Verifica che i DataFrame abbiano la stessa lunghezza
    if len(dates_df) != len(discounts_df):
        raise ValueError("I DataFrame 'dates_df' e 'discounts_df' devono avere la stessa lunghezza.")
    
    # Data di riferimento (come in MATLAB: datetime(2023, 02, 02))
    ref_date = date(2023, 2, 2)
    
    # Estrai e converte le date in datetime.date
    date_list = dates_df["Date"].apply(to_date).values
    
    # Estrai i discount factors e convertili in float (se fossero stringhe)
    discount_list = discounts_df["Discount Factor"].astype(float).values
    
    # Calcola la frazione d'anno con la convenzione 30/360 (in MATLAB era l'argomento 3)
    # Sostituisci 'mod.EU_30_360' con la tua costante/valore 
    # se la tua funzione yearfrac lo richiede diversamente.
    y_frac = np.array([yearfrac(ref_date, d, mod.EU_30_360) for d in date_list])
    
    # Calcola i tassi zero-coupon in percentuale:
    # zRate = -log(discount) / y_frac * 100
    z_rates = (-np.log(discount_list) / y_frac) * 100
    
    # Crea un DataFrame con le date e i tassi zero
    df_result = pd.DataFrame({"Zero Rate": z_rates})
    
    # Rimuovi la prima riga e reindicizza
    df_result = df_result.iloc[1:].reset_index(drop=True)

    return df_result

# Esempio d'uso
if __name__ == '__main__':
    # Esempio: creiamo due DataFrame di prova, 
    # ognuno con 5 righe (stesse dimensioni)
    
    # 1) DataFrame delle date
    dates_df = pd.DataFrame({
        "Date": [
            date(2024, 2, 2),
            date(2025, 2, 2),
            date(2026, 2, 2),
            date(2027, 2, 2),
            date(2028, 2, 2)
        ]
    })
    
    # 2) DataFrame dei discount factors (valori fittizi)
    discounts_df = pd.DataFrame({
        "Discount Factor": [0.95, 0.90, 0.85, 0.80, 0.75]
    })
    
    # Calcolo dei tassi zero
    df_zero = zeroRates(dates_df, discounts_df)
    print(df_zero)
