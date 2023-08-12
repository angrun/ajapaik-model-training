# Ajapaik model training

Core functionality for ajapaik-web images categories predictions

Commands to run service inside analytics host:

```
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=120 anna@ajapaik.ee -p 8022
cd ajapaik-model-training/

# Install all dependancies
pip install -r requirements.txt

# Run service
python3 manage.py runserver 8080
```

### Previous (archived) development projects for ajapaik learning

* https://github.com/angrun/ajapaik-learning
* https://github.com/angrun/ajapaik


### Wiki page 

https://github.com/angrun/ajapaik/wiki