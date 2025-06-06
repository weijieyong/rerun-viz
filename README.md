uv run simulate_ft_stream.py | uv run ft_sensor_stream.py --serve-web

uv run ft_sensor_csv.py --serve-web

uv run main.py

uv run serve-web.py

rerun --serve-web 
