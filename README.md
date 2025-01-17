# Large-scale Semantic Trails Global POI Dataset

## Usage

- Clone [Semantic Trails](https://github.com/D2KLab/semantic-trails) repository.
- Download [Semantic Trails](https://figshare.com/articles/dataset/Semantic_Trails_Datasets/7429076) dataset and extract it to `semantic-trails/downloads/`.
- Lookup the city relation ID from [OpenStreetMap](https://www.openstreetmap.org/).
- Download GeoJSON file from [Overpass Turbo](https://overpass-turbo.eu/#) for the desired city.

## Dataset Statistics

| City            | Users | Trails | POIs  | Check-ins | #train | #val  | #test |
| --------------- | :---: | :----: | :---: | :-------: | :----: | :---: | :---: |
| Beijing 🇨🇳       |  56   |  573   | 1127  |   1470    |  400   |  58   |  115  |
| Istanbul 🇹🇷      | 23700 | 216411 | 53812 |  544471   | 151487 | 21641 | 43283 |
| Jakarta 🇮🇩       | 8336  | 137396 | 76116 |  412100   | 96176  | 13740 | 27480 |
| Kuwait City 🇰🇼   | 9628  | 91658  | 17180 |  232706   | 64160  | 9166  | 18332 |
| Melbourne 🇦🇺     |  646  |  7864  | 7699  |   22050   |  5504  |  787  | 1573  |
| Moscow 🇷🇺        | 3993  | 39485  | 17822 |  105620   | 27639  | 3949  | 7897  |
| New York 🇺🇸      | 6929  | 92041  | 49218 |  272368   | 64428  | 9204  | 18409 |
| Petaling Jaya 🇲🇾 | 14308 | 180410 | 60158 |  506430   | 126287 | 18041 | 36082 |
| São Paulo 🇧🇷     | 5822  | 89689  | 38377 |  256824   | 62782  | 8969  | 17938 |
| Shanghai 🇨🇳      |  296  |  3636  | 4462  |   10491   |  2544  |  364  |  728  |
| Sydney 🇦🇺        |  740  | 10148  | 8986  |   29900   |  7103  | 1015  | 2030  |
| Tokyo 🇯🇵         |  764  |  5482  | 4725  |   13839   |  3836  |  549  | 1097  |

## Beijing 🇨🇳

<details>
<summary>Download Beijing GeoJSON</summary>

```sql
[out:json];
relation(912940);
out geom;
```
</details>

<details>
<summary>Preprocess STD Beijing Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/beijing/beijing_checkins.csv \
    --dataset_id w11wo/STD-Beijing-POI --private
```
</details>

## Istanbul 🇹🇷

<details>
<summary>Download Istanbul GeoJSON</summary>

```sql
[out:json];
relation(223474);
out geom;
```
</details>

<details>
<summary>Preprocess STD Istanbul Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/istanbul/istanbul_checkins.csv \
    --dataset_id w11wo/STD-Istanbul-POI --private
```
</details>

## Jakarta 🇮🇩

<details>
<summary>Download Jakarta GeoJSON</summary>
  
```sql
[out:json];
relation(6362934);
out geom;
```
</details>

<details>
<summary>Preprocess STD Jakarta Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/jakarta/jakarta_checkins.csv \
    --dataset_id w11wo/STD-Jakarta-POI --private
```
</details>

## Kuwait City 🇰🇼

<details>
<summary>Download Kuwait City GeoJSON</summary>

```sql
[out:json];
relation(305099);
out geom;
```
</details>

<details>
<summary>Preprocess STD Kuwait City Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/kuwait_city/kuwait_city_checkins.csv \
    --dataset_id w11wo/STD-Kuwait-City-POI --private
```
</details>

## Melbourne 🇦🇺

<details>
<summary>Download Melbourne GeoJSON</summary>

```sql
[out:json];
relation(4246124);
out geom;
```
</details>

<details>
<summary>Preprocess STD Melbourne Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/melbourne/melbourne_checkins.csv \
    --dataset_id w11wo/STD-Melbourne-POI --private
```
</details>

## Moscow 🇷🇺

<details>
<summary>Download Moscow GeoJSON</summary>

```sql
[out:json];
relation(2555133);
out geom;
```
</details>

<details>
<summary>Preprocess STD Moscow Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/moscow/moscow_checkins.csv \
    --dataset_id w11wo/STD-Moscow-POI --private
```
</details>

## New York 🇺🇸

<details>
<summary>Download New York GeoJSON</summary>

```sql
[out:json];
relation(175905);
out geom;
```
</details>

<details>
<summary>Preprocess STD New York Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/new_york/new_york_checkins.csv \
    --dataset_id w11wo/STD-New-York-POI --private
```
</details>

## Petaling Jaya 🇲🇾

<details>
<summary>Download Petaling Jaya GeoJSON</summary>

```sql
[out:json];
relation(8347386);
out geom;
```
</details>

<details>
<summary>Preprocess STD Petaling Jaya Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/petaling_jaya/petaling_jaya_checkins.csv \
    --dataset_id w11wo/STD-Petaling-Jaya-POI --private
```
</details>

## São Paulo 🇧🇷

<details>
<summary>Download São Paulo GeoJSON</summary>

```sql
[out:json];
relation(298285);
out geom;
```
</details>

<details>
<summary>Preprocess STD São Paulo Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/sao_paulo/sao_paulo_checkins.csv \
    --dataset_id w11wo/STD-Sao-Paulo-POI --private
```
</details>

## Shanghai 🇨🇳

<details>
<summary>Download Shanghai GeoJSON</summary>

```sql
[out:json];
relation(913067);
out geom;
```
</details>

<details>
<summary>Preprocess STD Shanghai Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/shanghai/shanghai_checkins.csv \
    --dataset_id w11wo/STD-Shanghai-POI --private
```
</details>

## Sydney 🇦🇺

<details>
<summary>Download Sydney GeoJSON</summary>

```sql
[out:json];
relation(5750005);
out geom;
```
</details>

<details>
<summary>Preprocess STD Sydney Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/sydney/sydney_checkins.csv \
    --dataset_id w11wo/STD-Sydney-POI --private
```
</details>

## Tokyo 🇯🇵

<details>
<summary>Download Tokyo GeoJSON</summary>

```sql
[out:json];
relation(1543125);
out geom;
```
</details>

<details>
<summary>Preprocess STD Tokyo Dataset</summary>

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
</details>

<details>
<summary>Create Next POI Dataset</summary>

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/tokyo/tokyo_checkins.csv \
    --dataset_id w11wo/STD-Tokyo-POI --private
```
</details>