# Large-scale Semantic Trails POI Dataset

## Usage

- Clone [Semantic Trails](https://github.com/D2KLab/semantic-trails) repository.
- Download [Semantic Trails](https://figshare.com/articles/dataset/Semantic_Trails_Datasets/7429076) dataset and extract it to `semantic-trails/downloads/`.
- Lookup the city relation ID from [OpenStreetMap](https://www.openstreetmap.org/).
- Download GeoJSON file from [Overpass Turbo](https://overpass-turbo.eu/#) for the desired city.

## Dataset Statistics

| City   | Users | Trajectories | POIs  | Check-ins | #train | #val  | #test |
| ------ | :---: | :----------: | :---: | :-------: | :----: | :---: | :---: |
| Sydney | 1029  |    10726     | 8986  |   31604   |  7507  | 1073  | 2146  |

## Example: Sydney

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
    --min_trails 2 --min_checkins 2
```

### Create Next POI Dataset

```sh
python src/create_next_poi_dataset.py \
    --checkins_file data/sydney/sydney_checkins.csv \
    --dataset_id w11wo/STD-Sydney-POI
```