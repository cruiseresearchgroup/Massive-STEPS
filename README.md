# Large-scale Semantic Trails Global POI Dataset

## Usage

- Clone [Semantic Trails](https://github.com/D2KLab/semantic-trails) repository.
- Download [Semantic Trails](https://figshare.com/articles/dataset/Semantic_Trails_Datasets/7429076) dataset and extract it to `semantic-trails/downloads/`.
- Lookup the city relation ID from [OpenStreetMap](https://www.openstreetmap.org/).
- Download GeoJSON file from [Overpass Turbo](https://overpass-turbo.eu/#) for the desired city.

## Dataset Statistics

| City        | Users | Trails | POIs  | Check-ins | #train | #val  | #test |
| ----------- | :---: | :----: | :---: | :-------: | :----: | :---: | :---: |
| Jakarta 🇮🇩   | 8336  | 137396 | 76116 |  412100   | 96176  | 13740 | 27480 |
| Moscow 🇷🇺    | 3993  | 39485  | 17822 |  105620   | 27639  | 3949  | 7897  |
| São Paulo 🇧🇷 | 5822  | 89689  | 38377 |  256824   | 62782  | 8969  | 17938 |
| Shanghai 🇨🇳  |  296  |  3636  | 4462  |   10491   |  2544  |  364  |  728  |
| Sydney 🇦🇺    |  740  | 10148  | 8986  |   29900   |  7103  | 1015  | 2030  |

## Jakarta 🇮🇩

### Download Jakarta GeoJSON

```sql
[out:json];
relation(6362934);
out geom;
```

### Preprocess STD Jakarta Dataset

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

### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/jakarta/jakarta_checkins.csv \
    --dataset_id w11wo/STD-Jakarta-POI --private
```

## Moscow 🇷🇺

### Download Moscow GeoJSON

```sql
[out:json];
relation(2555133);
out geom;
```

### Preprocess STD Moscow Dataset

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

### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/moscow/moscow_checkins.csv \
    --dataset_id w11wo/STD-Moscow-POI --private
```

## São Paulo 🇧🇷

### Download São Paulo GeoJSON

```sql
[out:json];
relation(298285);
out geom;
```

### Preprocess STD São Paulo Dataset

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

### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/sao_paulo/sao_paulo_checkins.csv \
    --dataset_id w11wo/STD-Sao-Paulo-POI --private
```

## Shanghai 🇨🇳

### Download Shanghai GeoJSON

```sql
[out:json];
relation(913067);
out geom;
```

### Preprocess STD Shanghai Dataset

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

### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/shanghai/shanghai_checkins.csv \
    --dataset_id w11wo/STD-Shanghai-POI --private
```

## Sydney 🇦🇺

### Download Sydney GeoJSON

```sql
[out:json];
relation(5750005);
out geom;
```

### Preprocess STD Sydney Dataset

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

### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/sydney/sydney_checkins.csv \
    --dataset_id w11wo/STD-Sydney-POI --private
```