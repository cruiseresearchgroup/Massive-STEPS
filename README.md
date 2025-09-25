# 🌏 Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins -- Dataset and Benchmarks

</div>

## 📖 Introduction

![Massive-STEPS](assets/world_map.png)

**Massive-STEPS** is a large-scale dataset of semantic trajectories intended for understanding POI check-ins. The dataset is derived from the [Semantic Trails Dataset](https://github.com/D2KLab/semantic-trails) and [Foursquare Open Source Places](https://huggingface.co/datasets/foursquare/fsq-os-places), and includes check-in data from 15 cities across 10 countries. The dataset is designed to facilitate research in various domains, including trajectory prediction, POI recommendation, and urban modeling. Massive-STEPS emphasizes the importance of geographical diversity, scale, semantic richness, and reproducibility in trajectory datasets.

## 🔢 Dataset

| City            | Users | Trails |  POIs  | Check-ins | #train  |  #val  | #test  |
| --------------- | :---: | :----: | :----: | :-------: | :-----: | :----: | :----: |
| Bandung 🇮🇩       | 3,377 | 55,333 | 29,026 |  161,284  | 113,058 | 16,018 | 32,208 |
| Beijing 🇨🇳       |  56   |  573   |  1127  |   1470    |   400   |   58   |  115   |
| Istanbul 🇹🇷      | 23700 | 216411 | 53812  |  544471   | 151487  | 21641  | 43283  |
| Jakarta 🇮🇩       | 8336  | 137396 | 76116  |  412100   |  96176  | 13740  | 27480  |
| Kuwait City 🇰🇼   | 9628  | 91658  | 17180  |  232706   |  64160  |  9166  | 18332  |
| Melbourne 🇦🇺     |  646  |  7864  |  7699  |   22050   |  5504   |  787   |  1573  |
| Moscow 🇷🇺        | 3993  | 39485  | 17822  |  105620   |  27639  |  3949  |  7897  |
| New York 🇺🇸      | 6929  | 92041  | 49218  |  272368   |  64428  |  9204  | 18409  |
| Palembang 🇮🇩     |  267  | 4,699  | 4,343  |  14,467   | 10,132  | 1,487  | 2,848  |
| Petaling Jaya 🇲🇾 | 14308 | 180410 | 60158  |  506430   | 126287  | 18041  | 36082  |
| São Paulo 🇧🇷     | 5822  | 89689  | 38377  |  256824   |  62782  |  8969  | 17938  |
| Shanghai 🇨🇳      |  296  |  3636  |  4462  |   10491   |  2544   |  364   |  728   |
| Sydney 🇦🇺        |  740  | 10148  |  8986  |   29900   |  7103   |  1015  |  2030  |
| Tangerang 🇮🇩     | 1,437 | 15,984 | 12,956 |  45,521   | 32,085  | 4,499  | 8,937  |
| Tokyo 🇯🇵         |  764  |  5482  |  4725  |   13839   |  3836   |  549   |  1097  |

### Dataset Construction

To reproduce the Massive-STEPS dataset, follow these steps:

1. Clone the [Semantic Trails](https://github.com/D2KLab/semantic-trails) repository.
    - This provides the necessary metadata, such as `cities.csv` and `categories.csv` that contains the mapping of POI categories and metadata about each city.
2. Download [Semantic Trails](https://figshare.com/articles/dataset/Semantic_Trails_Datasets/7429076) dataset from Figshare and extract it to `semantic-trails/downloads/`.
   - This provides `std_2013.csv` and `std_2018.csv` that contains the check-in data for 2013 and 2018, respectively. We will be using both subsets to create the Massive-STEPS dataset.
3. Download the city boundaries in GeoJSON format.
   - First, do a lookup of the target city's relation ID from [OpenStreetMap](https://www.openstreetmap.org/).
     - For example, search for "Beijing" and find the relation ID `912940`.
   - Then, using the related ID, download the geographical boundaries of the target city in GeoJSON file format from [Overpass Turbo](https://overpass-turbo.eu/#).
     - We recommend saving the GeoJSON file in the `data/<city_name>` directory.
4. Preprocess the check-in data using the `src/preprocess_std.py` script.
   - This script will filter the check-in data based on the geographical boundaries of the target city and create a CSV file containing the check-in data.
   - The script also filters out trajectories with less than 2 check-ins and users with less than 3 trajectories.
5. Create the next POI dataset using the `src/create_next_poi_dataset.py` script.
   - This script will create a dataset for the next POI recommendation task based on the filtered check-in data:
     - Each trajectory will be treated as a sequence of POIs and the last POI will be the target next POI.
     - Trajectories will also be converted into textual prompts for convenience.
   - The dataset will be split into training, validation, and test sets based on the user IDs.
   - The dataset will also be uploaded to Hugging Face Datasets with the specified dataset ID.

### End-to-End Examples

<details>
<summary>Bandung 🇮🇩</summary>

#### Download Bandung GeoJSON

```sql
[out:json];
relation(13290062);
out geom;
```

#### Preprocess Bandung Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/bandung/bandung_13290062_overpass.geojson \
    --output_dir data/bandung \
    --output_file bandung_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/bandung/beijing_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Bandung
```
</details>

<details>
<summary>Beijing 🇨🇳</summary>

#### Download Beijing GeoJSON

```sql
[out:json];
relation(912940);
out geom;
```

#### Preprocess Beijing Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/beijing/beijing_912940_overpass.geojson \
    --output_dir data/beijing \
    --output_file beijing_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/beijing/beijing_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Beijing
```
</details>

<details>
<summary>Istanbul 🇹🇷</summary>

#### Download Istanbul GeoJSON

```sql
[out:json];
relation(223474);
out geom;
```

#### Preprocess Istanbul Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/istanbul/istanbul_223474_overpass.geojson \
    --output_dir data/istanbul \
    --output_file istanbul_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/istanbul/istanbul_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Istanbul
```

</details>

<details>
<summary>Jakarta 🇮🇩</summary>

#### Download Jakarta GeoJSON
  
```sql
[out:json];
relation(6362934);
out geom;
```

#### Preprocess Jakarta Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/jakarta/jakarta_6362934_overpass.geojson \
    --output_dir data/jakarta \
    --output_file jakarta_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/jakarta/jakarta_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Jakarta
```

</details>

<details>
<summary>Kuwait City 🇰🇼</summary>

#### Download Kuwait City GeoJSON

```sql
[out:json];
relation(305099);
out geom;
```

#### Preprocess Kuwait City Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/kuwait_city/kuwait_city_305099_overpass.geojson \
    --output_dir data/kuwait_city \
    --output_file kuwait_city_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/kuwait_city/kuwait_city_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Kuwait-City
```

</details>

<details>
<summary>Melbourne 🇦🇺</summary>

#### Download Melbourne GeoJSON

```sql
[out:json];
relation(4246124);
out geom;
```

#### Preprocess Melbourne Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/melbourne/melbourne_4246124_overpass.geojson \
    --output_dir data/melbourne \
    --output_file melbourne_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/melbourne/melbourne_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Melbourne
```

</details>

<details>
<summary>Moscow 🇷🇺</summary>

#### Download Moscow GeoJSON

```sql
[out:json];
relation(2555133);
out geom;
```

#### Preprocess Moscow Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/moscow/moscow_2555133_overpass.geojson \
    --output_dir data/moscow \
    --output_file moscow_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/moscow/moscow_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Moscow
```

</details>

<details>
<summary>New York 🇺🇸</summary>

#### Download New York GeoJSON

```sql
[out:json];
relation(175905);
out geom;
```

#### Preprocess New York Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/new_york/new_york_175905_overpass.geojson \
    --output_dir data/new_york \
    --output_file new_york_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/new_york/new_york_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-New-York
```

</details>

<details>
<summary>Petaling Jaya 🇲🇾</summary>

#### Download Petaling Jaya GeoJSON

```sql
[out:json];
relation(8347386);
out geom;
```

#### Preprocess Petaling Jaya Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/petaling_jaya/petaling_jaya_8347386_overpass.geojson \
    --output_dir data/petaling_jaya \
    --output_file petaling_jaya_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/petaling_jaya/petaling_jaya_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Petaling-Jaya
```

</details>

<details>
<summary>Palembang 🇮🇩</summary>

#### Download Palembang GeoJSON

```sql
[out:json];
relation(10713145);
out geom;
```

#### Preprocess Palembang Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/palembang/palembang_10713145_overpass.geojson \
    --output_dir data/palembang \
    --output_file palembang_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/palembang/beijing_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Palembang
```
</details>

<details>
<summary>São Paulo 🇧🇷</summary>

#### Download São Paulo GeoJSON

```sql
[out:json];
relation(298285);
out geom;
```

#### Preprocess São Paulo Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/sao_paulo/sao_paulo_298285_overpass.geojson \
    --output_dir data/sao_paulo \
    --output_file sao_paulo_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/sao_paulo/sao_paulo_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Sao-Paulo
```

</details>

<details>
<summary>Shanghai 🇨🇳</summary>

#### Download Shanghai GeoJSON

```sql
[out:json];
relation(913067);
out geom;
```

#### Preprocess Shanghai Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/shanghai/shanghai_913067_overpass.geojson \
    --output_dir data/shanghai \
    --output_file shanghai_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/shanghai/shanghai_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Shanghai
```

</details>

<details>
<summary>Sydney 🇦🇺</summary>

#### Download Sydney GeoJSON

```sql
[out:json];
relation(5750005);
out geom;
```

#### Preprocess Sydney Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/sydney/sydney_5750005_overpass.geojson \
    --output_dir data/sydney \
    --output_file sydney_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/sydney/sydney_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Sydney
```

</details>

<details>
<summary>Tangerang 🇮🇩</summary>

#### Download Tangerang GeoJSON

```sql
[out:json];
relation(7641583);
out geom;
```

#### Preprocess Tangerang Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/tangerang/tangerang_7641583_overpass.geojson \
    --output_dir data/tangerang \
    --output_file tangerang_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/tangerang/beijing_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Tangerang
```
</details>

<details>
<summary>Tokyo 🇯🇵</summary>

#### Download Tokyo GeoJSON

```sql
[out:json];
relation(1543125);
out geom;
```

#### Preprocess Tokyo Dataset

```sh
python src/preprocess_std.py \
    --std_2013_file semantic-trails/downloads/std_2013.csv \
    --std_2018_file semantic-trails/downloads/std_2018.csv \
    --cities_file semantic-trails/cities.csv \
    --categories_file semantic-trails/categories.csv \
    --city_geo_json_file data/tokyo/tokyo_1543125_overpass.geojson \
    --output_dir data/tokyo \
    --output_file tokyo_checkins.csv \
    --min_checkins 2 --min_trails 3
```

#### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/tokyo/tokyo_checkins.csv \
    --dataset_id YOURUSERNAME/Massive-STEPS-Tokyo
```

</details>

## 📊 Benchmarks

### POI Recommendation

We also conducted extensive benchmarks on the Massive-STEPS dataset using various models for POI recommendation. The following table summarizes the results of our experiments, reported in Acc@1:

| **Model**    | **Bandung** | **Beijing** | **Istanbul** | **Jakarta** | **Kuwait City** | **Melbourne** | **Moscow** | **New York** | **Palembang** | **Petaling Jaya** | **São Paulo** | **Shanghai** | **Sydney** | **Tangerang** | **Tokyo** |
| ------------ | :---------: | :---------: | :----------: | :---------: | :-------------: | :-----------: | :--------: | :----------: | :-----------: | :---------------: | :-----------: | :----------: | :--------: | :-----------: | :-------: |
| **FPMC**     |    0.048    |    0.000    |    0.026     |    0.029    |      0.021      |     0.062     |   0.059    |    0.032     |     0.102     |       0.026       |     0.030     |    0.084     |   0.075    |     0.104     |   0.176   |
| **RNN**      |    0.062    |    0.085    |    0.077     |    0.049    |      0.087      |     0.059     |   0.075    |    0.061     |     0.049     |       0.064       |     0.097     |    0.055     |   0.080    |     0.087     |   0.133   |
| **LSTPM**    |    0.110    |    0.127    |    0.142     |    0.099    |      0.180      |     0.091     |   0.151    |    0.099     |     0.114     |       0.099       |     0.158     |    0.099     |   0.141    |     0.154     |   0.225   |
| **DeepMove** |    0.107    |    0.106    |    0.150     |    0.103    |      0.179      |     0.083     |   0.143    |    0.097     |     0.084     |       0.112       |     0.160     |    0.085     |   0.129    |     0.145     |   0.201   |
| **GETNext**  |    0.179    |    0.433    |    0.146     |    0.155    |      0.175      |     0.100     |   0.175    |    0.134     |     0.158     |       0.139       |     0.202     |    0.115     |   0.181    |     0.224     |   0.180   |
| **STHGCN**   |    0.219    |  **0.453**  |  **0.241**   |  **0.197**  |    **0.225**    |   **0.168**   | **0.223**  |  **0.146**   |   **0.246**   |     **0.174**     |   **0.250**   |  **0.193**   | **0.227**  |   **0.293**   | **0.250** |
| **UniMove**  |    0.007    |    0.036    |    0.015     |    0.004    |      0.023      |     0.008     |   0.009    |    0.004     |     0.009     |       0.008       |     0.002     |    0.000     |   0.015    |     0.001     |   0.032   |

### Zero-shot POI Recommendation

We also conducted zero-shot POI recommendation experiments on the Massive-STEPS dataset. The following table summarizes the results of our experiments, reported in Acc@1:

| **Method**   | **Model**                               | **Bandung** | **Beijing** | **Istanbul** | **Jakarta** | **Kuwait City** | **Melbourne** | **Moscow** | **New York** | **Palembang** | **Petaling Jaya** | **São Paulo** | **Shanghai** | **Sydney** | **Tangerang** | **Tokyo** |
| ------------ | --------------------------------------- | :---------: | :---------: | :----------: | :---------: | :-------------: | :-----------: | :--------: | :----------: | :-----------: | :---------------: | :-----------: | :----------: | :--------: | :-----------: | :-------: |
| **LLM-Mob**  | **gemini-2.0-flash**                    |    0.105    |    0.115    |    0.080     |    0.100    |      0.095      |     0.060     |   0.130    |    0.095     |     0.135     |       0.090       |     0.130     |    0.055     |   0.060    |     0.155     |   0.140   |
|              | **Qwen2.5-7B-Instruct-AWQ**             |    0.060    |    0.058    |    0.035     |    0.105    |      0.080      |     0.030     |   0.090    |    0.070     |     0.075     |       0.030       |     0.090     |    0.040     |   0.035    |     0.095     |   0.110   |
|              | **Meta-Llama-3.1-8B-Instruct-AWQ-INT4** |    0.010    |    0.000    |    0.020     |    0.055    |      0.030      |     0.010     |   0.030    |    0.025     |     0.005     |       0.010       |     0.030     |    0.005     |   0.020    |     0.020     |   0.005   |
|              | **gemma-2-9b-it-AWQ-INT4**              |    0.070    |    0.115    |    0.075     |    0.105    |      0.080      |     0.055     |   0.100    |    0.070     |     0.095     |       0.055       |     0.085     |    0.050     |   0.030    |     0.145     |   0.145   |
| **LLM-ZS**   | **gemini-2.0-flash**                    |    0.095    |    0.058    |    0.090     |    0.110    |      0.080      |     0.065     |   0.125    |    0.080     |     0.130     |       0.110       |     0.150     |    0.065     |   0.060    |     0.145     |   0.160   |
|              | **Qwen2.5-7B-Instruct-AWQ**             |    0.055    |    0.038    |    0.040     |    0.065    |      0.050      |     0.040     |   0.080    |    0.050     |     0.050     |       0.045       |     0.095     |    0.045     |   0.045    |     0.100     |   0.120   |
|              | **Meta-Llama-3.1-8B-Instruct-AWQ-INT4** |    0.045    |    0.077    |    0.040     |    0.045    |      0.060      |     0.040     |   0.080    |    0.055     |     0.070     |       0.030       |     0.030     |    0.060     |   0.040    |     0.080     |   0.110   |
|              | **gemma-2-9b-it-AWQ-INT4**              |    0.065    |    0.096    |    0.045     |    0.105    |      0.070      |     0.050     |   0.080    |    0.075     |     0.060     |       0.065       |     0.075     |    0.050     |   0.045    |     0.100     |   0.110   |
| **LLM-Move** | **gemini-2.0-flash**                    |  **0.225**  |    0.096    |  **0.205**   |  **0.295**  |    **0.220**    |   **0.225**   |   0.220    |  **0.235**   |   **0.260**   |     **0.210**     |   **0.285**   |  **0.170**   | **0.230**  |   **0.200**   | **0.250** |
|              | **Qwen2.5-7B-Instruct-AWQ**             |    0.100    |  **0.192**  |    0.175     |    0.115    |      0.160      |     0.110     | **0.230**  |    0.120     |     0.130     |       0.135       |     0.155     |    0.095     |   0.125    |     0.175     | **0.250** |
|              | **Meta-Llama-3.1-8B-Instruct-AWQ-INT4** |    0.030    |    0.058    |    0.015     |    0.015    |      0.010      |     0.040     |   0.005    |    0.035     |     0.010     |       0.040       |     0.045     |    0.020     |   0.055    |     0.000     |   0.030   |
|              | **gemma-2-9b-it-AWQ-INT4**              |    0.175    |    0.096    |    0.100     |    0.235    |      0.120      |     0.115     |   0.110    |    0.115     |     0.210     |       0.175       |     0.195     |    0.105     |   0.125    |     0.125     |   0.130   |

### Spatiotemporal Classification and Reasoning

We conducted spatiotemporal classification and reasoning experiments on the Massive-STEPS dataset. Specifically, we evaluated the zero-shot performance of LLMs to classify whether a check-in trajectory ended at a weekend (Saturday or Sunday) or a weekday (Monday to Friday). The following table summarizes the results of our experiments, reported in accuracy:

| **Model**            | **Bandung** | **Beijing** | **Istanbul** | **Jakarta** | **Kuwait City** | **Melbourne** | **Moscow** | **New York** | **Palembang** | **Petaling Jaya** | **São Paulo** | **Shanghai** | **Sydney** | **Tangerang** | **Tokyo** |
| -------------------- | :---------: | :---------: | :----------: | :---------: | :-------------: | :-----------: | :--------: | :----------: | :-----------: | :---------------: | :-----------: | :----------: | :--------: | :-----------: | :-------: |
| **gemini-2.0-flash** |  **0.635**  |    0.615    |  **0.715**   |  **0.650**  |    **0.765**    |   **0.635**   |   0.740    |  **0.620**   |   **0.670**   |     **0.610**     |   **0.730**   |  **0.600**   | **0.550**  |   **0.635**   |   0.510   |
| **gpt-4o-mini**      |    0.625    |    0.538    |    0.610     |    0.610    |      0.430      |   **0.635**   | **0.745**  |    0.600     |     0.645     |       0.590       |     0.645     |    0.565     |   0.545    |     0.600     |   0.495   |
| **gpt-4.1-mini**     |    0.585    |  **0.673**  |    0.615     |    0.600    |      0.690      |     0.585     | **0.745**  |    0.595     |     0.605     |       0.575       |     0.700     |    0.565     |   0.515    |     0.620     |   0.550   |
| **gpt-5-nano**       |    0.570    |    0.635    |    0.535     |    0.530    |      0.470      |     0.500     |   0.635    |    0.580     |     0.560     |       0.565       |     0.680     |    0.465     |   0.440    |     0.520     | **0.580** |

## 🧪 Replicate Experiments

### Install Dependencies

Our experiments rely on the following libraries:

- [AgentMove](https://github.com/tsinghua-fib-lab/agentmove)
- [GETNext](https://github.com/songyangme/GETNext)
- [STHGCN](https://github.com/alipay/Spatio-Temporal-Hypergraph-Model/)
- [LibCity](https://github.com/LibCity/Bigscity-LibCity)
- [UniMove](https://github.com/tsinghua-fib-lab/UniMove)

To reproduce our experiments, clone this repository and its submodules:

```sh
git clone https://github.com/YOURUSERNAME/Massive-STEPS
cd Massive-STEPS
git submodule update
```

where the submodules are our forks of the original repositories with some modifications to support Massive-STEPS. Each of the submodules has its own updated dependencies, as listed in their respective `requirements.txt` files.

Because the submodules are not installed in the same directory as the main repository, we need to create a softlink for each submodule to point to the `data/` directory. You can do this by running the following commands:

```sh
ln -s $(pwd)/data $(pwd)/GETNext/data
ln -s $(pwd)/data $(pwd)/Spatio-Temporal-Hypergraph-Model/data
ln -s $(pwd)/data $(pwd)/Bigscity-LibCity/data
```

### POI Recommendation

#### FPMC, RNN, LSTPM, DeepMove

Our experiments on FPMC, RNN, LSTPM, and DeepMove are based on the implementations of [LibCity](https://github.com/LibCity/Bigscity-LibCity). To run the experiments, you can use the following command:

```sh
city=beijing # or istanbul, jakarta, etc.
python convert_std.py \
    --city $city \
    --path data \
    --output_path raw_data

for model in FPMC RNN DeepMove; do
    python run_model.py \
        --task traj_loc_pred \
        --model $model \
        --dataset std_$city \
        --config libcity/config/data/STDDataset
done

for model in LSTPM; do
    python run_model.py \
        --task traj_loc_pred \
        --model $model \
        --dataset std_$city \
        --config libcity/config/data/STDDataset$model
done
```

This will run the experiments for each model on the specified city. The results will be saved in the `libcity/cache/{exp}/evaluate_cache/` directory.

You can refer to `Bigscity-LibCity/run_train.sh` for the full list of models and cities used in our experiments. The script will run the experiments for each model on all cities in the Massive-STEPS dataset.

#### GETNext

To run the GETNext experiments, you can use the following command:

```bash
city=beijing # or istanbul, jakarta, etc.
python build_graph.py \
    --csv_path data/$city/${city}_checkins_train.csv \
    --output_dir data/$city

python train.py \
    --data-train data/$city/${city}_checkins_train.csv \
    --data-val data/$city/${city}_checkins_test.csv \
    --data-adj-mtx data/$city/graph_A.csv \
    --data-node-feats data/$city/graph_X.csv \
    --time-units 48 --timestamp_column timestamp \
    --poi-embed-dim 128 --user-embed-dim 128 \
    --time-embed-dim 32 --cat-embed-dim 32 \
    --node-attn-nhid 128 \
    --transformer-nhid 1024 \
    --transformer-nlayers 2 --transformer-nhead 2 \
    --batch 16 --epochs 200 --name $city \
    --workers 12 --exist-ok \
    --lr 0.001
```

This will run the GETNext experiments on the specified city. The results will be saved in the `runs/train/{city}/` directory.

You can refer to `GETNext/run_train.sh` for the full list of cities used in our experiments. The script will run the GETNext experiments for each city in the Massive-STEPS dataset, with the hyperparameters we used in our experiments.

#### STHGCN

To run the STHGCN experiments, you can use the following command:

```bash
city=beijing # or istanbul, jakarta, etc.
python run.py -f std_conf/$city.yml
```

This will run the STHGCN experiments on the specified city. The results will be saved in the `log/{exp}/{city}/` directory.

You can refer to `Spatio-Temporal-Hypergraph-Model/run_train.sh` for the full list of cities used in our experiments.

#### UniMove

To run the UniMove experiments, you can use the following command:

```bash
city=beijing # or istanbul, jakarta, etc.

python preprocess_massive_steps.py --city $city

python main.py \
    --device cuda:0 \
    --city $city \
    --target_city $city \
    --train_root traj_dataset/massive_steps/train \
    --val_root traj_dataset/massive_steps/val \
    --test_root traj_dataset/massive_steps/test \
    --B 4
```

This will train UniMove from scratch on the specified city. The results will be saved in the `models/{city}/` directory.

You can refer to `UniMove/run_train.sh` for the full list of cities used in our experiments.

### Zero-shot POI Recommendation

To run a zero-shot POI recommendation experiment, you can use the following command:

```bash
model=gemini-2.0-flash # or Qwen2.5-7B-Instruct-AWQ, etc.
prompt_typ=llmzs # or llmmob, llmmove
city=Beijing # Istanbul, Jakarta, etc.
city_key=$(echo "$city" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

python src/run_next_poi_llm.py \
    --dataset_name YOURUSERNAME/Massive-STEPS-$city \ # Hugging Face dataset ID
    --num_users 200 --num_historical_stays 15 \ # number of test users and historical stays
    --prompt_type $prompt_type \ 
    --model_name $model \
    --checkins_file data/$city_key/${city_key}_checkins.csv # path to the check-in data
```

This will run the zero-shot POI recommendation experiment using the specified model and prompt type on the specified city. The results will be saved in the `results/{dataset_name}/{model_name}/{prompt_type}/` directory.

You can refer to the `run_next_poi_llm.sh` script to see the full list of models and prompt types used in our experiments. The script will run the zero-shot POI recommendation experiment for each model and prompt type on all cities in the Massive-STEPS dataset.

### Spatiotemporal Classification and Reasoning

To run a spatiotemporal classification and reasoning experiment, you can use the following command:

```bash
model=gemini-2.0-flash # or gpt-4o-mini, etc.
city=Beijing # Istanbul, Jakarta, etc.
city_key=$(echo "$city" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

python src/run_classify_day_llm.py \
    --dataset_name YOURUSERNAME/Massive-STEPS-$city \ # Hugging Face dataset ID
    --num_users 200  \ # number of test users
    --prompt_type st_day_classification \
    --model_name $model \
    --city $city \
    --checkins_file data/$city_key/${city_key}_checkins.csv # path to the check-in data
```

#### Hosting LLMs

To host open-source LLMs, we recommend using [vLLM](https://github.com/vllm-project/vllm) which is a high-performance, memory-efficient, and easy-to-use library for serving large language models. Once installed, you can run the following command to start the server:

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --quantization awq
```

and modify `VLLM_API_BASE_URL` and `VLLM_API_KEY` environment variables accordingly, pointing to the server's address.

## 📜 Acknowledgement

Our work is based on the following repositories:

- [AgentMove](https://github.com/tsinghua-fib-lab/agentmove)
- [GETNext](https://github.com/songyangme/GETNext)
- [STHGCN](https://github.com/alipay/Spatio-Temporal-Hypergraph-Model/)
- [LibCity](https://github.com/LibCity/Bigscity-LibCity)
- [UniMove](https://github.com/tsinghua-fib-lab/UniMove)
