# Gerekli Kütüphane ve Fonksiyonlar

#!pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

#Verinin Okunması ve Kopyasının oluşturulması

df_ = pd.read_csv(r"C:\Users\lenovo\Desktop\vbo-ml\hafta3\ödevler\FLO_RFM_ANALIZI\FLO_RFM_Analizi\flo_data_20K.csv")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

#Adım 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımladık.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

#Sipariş ve fatura değerine ait aykırı değerleri baskıladık.

num_values_col = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in num_values_col:
    replace_with_thresholds(df, col)

#Genel veriler ile analiz yapabilmek için online ve offline kanalları birleştirdik.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Tarihler analiz için önem arzettiğindne tarih barındıran değişkenleri "datetime64[ns]" tipine çevirdik.

for i in df.columns:
    if "date" in i:
       df[i] = df[i].astype("datetime64[ns]")

# Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak aldık.

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

#customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni
# bir cltv dataframe'i oluşturduk.

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süreönce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)cltv_df[(cltv_df['frequency'] > 1)]
# monetary: satın alma başına ortalama kazanç

cltv_df = pd.DataFrame()
cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_weekly": ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
             "frequency": df["order_num_total"],
             "monetary_cltv_avg": df["customer_value_total"] / df["order_num_total"]})

cltv_df.head()
cltv_df.reset_index()

#BG/NBD, Gamma-Gamma Modelini fit ettik.
#penalizer_coef = 0,001 parametrelerin bulunması aşamasında katsayılara uygulanacak ceza katsayısıdır

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# 3 ay ve 6 ay içerisinde müşterilerden beklenen satın almaları tahmin edip ve exp_sales_3_month olarak cltv dataframe'ine ekledik.

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

bgf.predict(4 * 6,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly']).sum()

cltv_df["expected_sales_3_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

cltv_df.head()

#Gamma-Gamma modelini fit ettik. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
#dataframe'ine ekledik.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()

#Gamma-Gamma modeli ile 6 aylık CLTV hesapladık ve cltv ismiyle dataframe'e ekledik.
# Cltv değeri en yüksek 20 kişiyi gözlemledik.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6aylık
                                   freq="W",  # hafta
                                   discount_rate=0.05)

cltv_df["cltv"] = cltv

cltv_df.sort_values(by="cltv", ascending=False).head(20)

#CLTV Değerine Göre Segmentlerin Oluşturulması kısımına geldik.
#6 aylık standartlaştırılmış CLTV'ye göre tüm müşterileri 4 gruba (segmente) ayırdık.

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

#CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır? Kontrol edelim.

cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})

#c ve b segmentlerının degerlerı bırbırne cok yakın oldugundan dolayı,
# segment sayısını 3'e düşürmek marka tarafından düşünülebilir. Eğer operasyon
# tarafında maliyetleri düşürecekse segment sayısını azaltmak markaya önerilebilir.
# Müşterileri 3'e böldüğümüz yeni bir segment yapısı tasarladık.

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 3, labels=["C", "B", "A"])
cltv_df.head()
cltv_df.groupby("segment").agg(
    {"count", "mean", "sum"})
