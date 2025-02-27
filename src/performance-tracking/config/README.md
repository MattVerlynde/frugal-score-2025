# Configuration for energy consumption tracking

Cette page résume la procédure d'installation du pipeline de lecture, écriture, sauvegarde et visualisation des données internes du hardware. Celle-ci est basée sur le pipeline **Telegraf-InfluxDB-Grafana** (TIG). Elle s'inspire largement sur le tutoriel accessible en ligne à [https://domopi.eu/tig-le-trio-telegraf-influxdb-grafana-pour-surveiller-vos-equipements/](https://domopi.eu/tig-le-trio-telegraf-influxdb-grafana-pour-surveiller-vos-equipements/).
Cette page présente également la procédure d'interrogation de la base de données **InfluxDB** après exécution d'un programme python, via l'exécution d'un script bash.

<!--more-->

## Présentation du pipeline


Le plugin Telegraf, produit par InfluxDB permet la collection des données du hardware de l'ordinateur en temps réel, ainsi que son formatage. Le plugin ZWave-JS UI collecte les données de la prise connectée, et les transfère à Telegraf via un plugin Mosquitto.
InfluxDB permet le stockage de ces données en séries temporelles, et constitue la base de donnée qui est interrogée au sein du pipeleine; Grafana est un outil de visualisation et d'analyse des données lues la base de données de InfluxDB.

## Téléchargement de TIG

> Ce tutoriel se concentre sur l'installation de pipeline via docker, et nécessite donc son installation préalable.

Commençons par télécharger les images docker de trois plugins constituant le pipeline.

```shell
docker pull telegraf
docker pull influxdb
docker pull grafana/grafana-oss
```
Nous allons utiliser la commande `docker compose` afin de réaliser notre pipeline. Construisons alors le fichier de construction `docker-compose.yml`. Configurons le port d'entrée de Grafana selon notre choix. Dans cet exemple, nous avons choisi le port `9090`.

```yaml
version: "3.8"
services:
  influxdb:
    image: influxdb
    container_name: influxdb
    restart: always
    ports:
      - 8086:8086
    hostname: influxdb
    environment:
      INFLUX_DB: $INFLUX_DB  # nom de la base de données créée à l'initialisation d'InfluxDB
      INFLUXDB_USER: $INFLUXDB_USER  # nom de l'utilisateur pour gérer cette base de données
      INFLUXDB_USER_PASSWORD: $INFLUXDB_USER_PASSWORD  # mot de passe de l'utilisateur pour gérer cette base de données
      DOCKER_INFLUXDB_INIT_MODE: $DOCKER_INFLUXDB_INIT_MODE
      DOCKER_INFLUXDB_INIT_USERNAME: $DOCKER_INFLUXDB_INIT_USERNAME
      DOCKER_INFLUXDB_INIT_PASSWORD: $DOCKER_INFLUXDB_INIT_PASSWORD
      DOCKER_INFLUXDB_INIT_ORG: $DOCKER_INFLUXDB_INIT_ORG
      DOCKER_INFLUXDB_INIT_BUCKET: $DOCKER_INFLUXDB_INIT_BUCKET
      DOCKER_INFLUXDB_INIT_RETENTION: $DOCKER_INFLUXDB_INIT_RETENTION
      DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: $DOCKER_INFLUXDB_INIT_ADMIN_TOKEN
    volumes:
      - ./influxdb:/var/lib/influxdb  # volume pour stocker la base de données InfluxDB

  telegraf:
    image: telegraf
    depends_on:
      - influxdb  # indique que le service influxdb est nécessaire
    container_name: telegraf
    restart: always
    links:
      - influxdb:influxdb
    tty: true
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # nécessaire pour remonter les données du démon Docker
      - ./telegraf/telegraf.conf:/etc/telegraf/telegraf.conf  # fichier de configuration de Telegraf
      - /:/host:ro # nécessaire pour remonter les données de l'host (processes, threads...)

  grafana:
    image: grafana/grafana-oss
    depends_on:
      - influxdb  # indique que le service influxdb est nécessaire
    container_name: grafana
    restart: always
    ports:
      - 9090:3000  # port pour accéder à l'interface web de Grafana
    links:
      - influxdb:influxdb
    environment:
      GF_INSTALL_PLUGINS: "grafana-clock-panel,\
                          grafana-influxdb-08-datasource,\
                          grafana-kairosdb-datasource,\
                          grafana-piechart-panel,\
                          grafana-simple-json-datasource,\
                          grafana-worldmap-panel"
      GF_SECURITY_ADMIN_USER: $GF_SECURITY_ADMIN_USER  # nom de l'utilisateur créé par défaut pour accéder à Grafana
      GF_SECURITY_ADMIN_PASSWORD: $GF_SECURITY_ADMIN_PASSWORD  # mot de passe de l'utilisateur créé par défaut pour accéder à Grafana
    volumes:
      - ./grafana:/var/lib/grafana-oss
```

> Ce fichier est disponible: [tig/docker-compose.yml](tig/docker-compose.yml).


Ce fichier est construit en dépendance d'un fichier de contenance des variables d'environnement. Construisonsce fichier `.env` avec les valeurs de ces variables dans le même dossier.

```yaml
INFLUX_DB=telegraf
INFLUXDB_USER=telegraf_user
INFLUXDB_USER_PASSWORD=telegraf_password
GF_SECURITY_ADMIN_USER=grafana_user
GF_SECURITY_ADMIN_PASSWORD=grafana_password
DOCKER_INFLUXDB_INIT_MODE=setup
DOCKER_INFLUXDB_INIT_USERNAME=telegraf_user
DOCKER_INFLUXDB_INIT_PASSWORD=telegraf_password
DOCKER_INFLUXDB_INIT_ORG=telegraf_org
DOCKER_INFLUXDB_INIT_BUCKET=telegraf_bucket
DOCKER_INFLUXDB_INIT_RETENTION=365d
DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=telegraf_token
```

> Ce fichier est disponible: [tig/.env](tig/.env).

Nous allons maintenant configurer les paramètres de Telegraf. Dans le shell, exécutez la commane suivante.

```shell
mkdir telegraf
docker run --rm telegraf telegraf config > telegraf/telegraf.conf
```

Cette commande nous a permi de créer un fichier de configuration par défaut de Telegraf, que nous alons alors modifier pour notre projet.

```squidconf

# Configuration for telegraf agent
[agent]
  
  [...]

  ## Override default hostname, if empty use os.Hostname()
  hostname = "telegraf"
  ## If set to true, do no set the "host" tag in the telegraf agent.
  omit_hostname = false

[...]

###############################################################################
#                            OUTPUT PLUGINS                                   #
###############################################################################


# # Configuration for sending metrics to InfluxDB 2.0
[[outputs.influxdb_v2]]
#   ## The URLs of the InfluxDB cluster nodes.
#   ##
#   ## Multiple URLs can be specified for a single cluster, only ONE of the
#   ## urls will be written to each interval.
#   ##   ex: urls = ["https://us-west-2-1.aws.cloud2.influxdata.com"]
   urls = ["http://influxdb:8086"]
#
#   ## Token for authentication.
   token = "telegraf_token"
#
#   ## Organization is the name of the organization you wish to write to.
   organization = "telegraf_org"
#
#   ## Destination bucket to write into.
   bucket = "telegraf_bucket"

   [...]

[...]

[[outputs.influxdb]]
#   ## The full HTTP or UDP URL for your InfluxDB instance.
#   ##
#   ## Multiple URLs can be specified for a single cluster, only ONE of the
#   ## urls will be written to each interval.
#   # urls = ["unix:///var/run/influxdb.sock"]
#   # urls = ["udp://127.0.0.1:8089"]
#   # urls = ["http://127.0.0.1:8086"]
   urls = ["http://influxdb:8086"]

   [...]

#   ## HTTP Basic Auth
   username = "telegraf_user"
   password = "telegraf_password"

   [...]

[...]

[[inputs.docker]]
#   ## Docker Endpoint
#   ##   To use TCP, set endpoint = "tcp://[ip]:[port]"
#   ##   To use environment variables (ie, docker-machine), set endpoint = "ENV"
   endpoint = "unix:///var/run/docker.sock"

   [...]

[...]

# # Monitor process cpu and memory usage
[[inputs.procstat]]
   pattern = ".*"
   fieldpass = ["cpu_time_system", "cpu_time_user", "cpu_usage", "memory_*", "num_threads", "*pid"]
   pid_finder = "native"
   pid_tag = true

   [...]

[...]

# # Read metrics about temperature
[[inputs.temp]]

[...]

```
Effectuez alors les modifications suivantes :
* Dans `[agent]`, la variable `hostname` comme la variable d'environnement `INFLUX_DB`, ici `"telegraf"`

* Décommenter `[[outputs.influxdb_v2]]`, la variable `urls` comme `["http://influxdb:8086"]`, et les variables `token`, `organization` et `bucket` comme `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN`, `DOCKER_INFLUXDB_INIT_ORG` et `DOCKER_INFLUXDB_INIT_BUCKET`, ici de valeur `"telegraf_token"`, `"telegraf_org"` et `"telegraf_bucket"`.

* Commenter `[[outputs.file]]`. Garder un fichier de sauvegarde au sein de Telegraf ne sera pas utile en complément de la base InfluxDB.

* Décommenter `[[outputs.influxdb]]`, la variable `urls` comme `["http://influxdb:8086"]`, et les variables `username` et `password` comme `DOCKER_INFLUXDB_INIT_USERNAME` et `DOCKER_INFLUXDB_INIT_PASSWORD`, ici de valeur `"telegraf_user"` et `"telegraf_password"`.

* Décommenter `[[inputs.docker]]`, la variable `endpoint` comme `"unix:///var/run/docker.sock"`.

* Décommenter `[[inputs.procstat]]`, définir la variable `pattern` comme `".*"` pour récupérer les données de tous les processes, `fieldpass` comme les variables que l'on souhaite récupérer, ici `["cpu_time_system", "cpu_time_user", "cpu_usage", "memory_*", "num_threads", "*pid"]`, `pid_finder` comme `"native"` afin d'accéder aux données de l'hôte hors du container, et `pid_tag` comme `true` afin de conserver l'identifiant des processes,

* Décommenter `[[inputs.temp]]` pour obtenir les données de températures du CPU et de la NVME.

> Ce fichier est disponible: [tig/telegraf/telegraf.conf](tig/telegraf/telegraf.conf).

Nous pouvons alors ensuite créer les conteneurs.

```shell
docker compose up -d
```

Vérifiez que les conteneurs ont bien été créés avec la commande suivante:

```shell
docker ps
```

Si nous avons bien créé les conteneurs, vous pouvons accéder à l'interface de Grafana sur le port choisi, ici `http://localhost:9090`.

> Nous pouvons également accéder à l'interface de InfluxDB via le port configuré dans le fichier `docker-compose.yml`, ici à l'adresse `http://localhost:8086`.

Nous pouvons alors passer à la configuration de Grafana.

## Configuration de Grafana

Sur la page d'accueil de Grafana, connectons nous avec l'identifiant et le mot de passe configuré dans le fichier `.env`. Dans notre exemple, nous avons `grafana_user` et `grafana_password`.

[Page d'accueil de Grafana](screenshots_config/grafana_welcome.png)

Configurons la source des données dans l'onger Data source, et choisissons InfluxDB comme tyope de source.

[Sélection de la source des données (InfluxDB)](screenshots_config/grafana_select_influx.png)

Configurons maintenant la source des données avec le port de InfluxDB, et choisissons `FLUX` comme langage d'interrogation de la base.

[Sélection du nom et du port](screenshots_config/grafana_set_datasource1.png)

Ajoutons ensuite les identifiants de connexion à la base de données avec ceux choisis dans le fichier `.env`.

[Sélection ajout des identifiant](screenshots_config/grafana_set_datasource2.png)

Importons ensuite un dashboard de visualisation des données compatible avec nos configurations. Nous pouvons choisir un dashboard compatible en ligne, mais le dashboard correspondant à l'identifiant `15650` convient à notre exemple.

[Importation du dashboard](screenshots_config/grafana_import_dashb1.png)

Choisissons la source que nous avons configuré avant d'importer.

[Sélection de la source](screenshots_config/grafana_import_dashb2.png)

Enfin, choisissons les paramètres du dashboard correspondant à nos données, ici le nom du bucket que nous avons configuré.

[Sélection des paramètres](screenshots_config/grafana_import_dashb3.png)

Nous pouvons ensuite modifier plus finement les affichages du dashboard selon nos objectifs, en modifiant leurs paramètres ou les queries associées (en respectant le langage d'écriture Flux).

Une autre possibilité pour importer un dashboard est d'importer le fichier associé au forma `.json`. Le fichier configuré que nous pouvons importer est disponible sur la: `grafana_dashboard_template_07052024.json`.

Le dashboard créé lors de notre projet est accessible sur [ce lien](http://localhost:9090/d/edh1jtjp0b4lcb/38860f63-1c6f-5143-a2a7-cfc431003966?orgId=1&from=1715083321958&to=1715084221958).

## Interrogation de la base

Afin d'interrogée la base de données **InfluxDB** créée précédemment, nous utilisons un script bash exécute un fichier python présenté en argument, puis interroge la base de donnée afin de récolter les données enregistrée sur la période d'exécution du fichier python.

> Ce fichier est disponible: [get_metrics.sh](get_metrics.sh).

L'exécution de ce fichier est réalisée selon la commande suivante :

```bash
bash get_metrics.sh -f [python.file] (-p) (-P [pid])
```
* le drapeau `-f` est obligatoire et précède le nom du fichier python à exécuter
* le drapeau `-P` est facultatif : si renseigné, les données récoltées seront celles associées au process dont l'identifiant est disponible dans un fichier `python_process.pid` sur la période d'exécution du fichier python
* le drapeau `-p` est facultatif : si renseigné, les données récoltées seront celles associées au process dont l'identifiant est renseigné comme argument dans la commande,
* si les drapeaux `-p` et `-P` ne sont pas renseignés, l'ensemble des données de la base sur la période d'exécution du fichier python sera récoltée.

Les données récoltées sont enregistrées dans un fichier `metrics_output`.

Le fichier de récolte de donnée s'organise ainsi :

```bash
#!/bin/bash

#Récolte des arguments en entrée
while getopts 'f:p:P:' OPTION; do
  case "$OPTION" in
    f)
      name_file="$OPTARG"
      ;;
    P) 
      by_pid=true
      get_pid=true
      ;;
    p) 
      by_pid=true
      get_pid=false
      npid="$OPTARG"
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
  esac
done
: ${name_file:?Missing -f}

#Relevé du premier temps avant exécution du fichier python
t1=$(date -u +%Y-%m-%dT%T.%9NZ)
echo "*************************************************"
echo "Time start: $t1"
echo "*************************************************"
echo "Running python script: $name_file"
echo "*************************************************"

#Exécution du fichier python
python3 $name_file

#Relevé du second temps après exécution du fichier python
t2=$(date -u +%Y-%m-%dT%T.%9NZ)

echo "*************************************************"
echo "Time stop: $t2"
echo "*************************************************"


if [ "$by_pid" = true ]; then
  if [ "$get_pid" = true ]; then
    # Récolte du pid si non renseigné en entrée 
    npid=$(cat python_process.pid)
  fi
  echo "Process ID: ${npid}"
  # Construction de la query sur le process
  query="data=from(bucket: \"telegraf_bucket\")
    |> range(start: ${t1}, stop: ${t2})
    |> filter(fn: (r) => r[\"_measurement\"] == \"procstat\")
    |> filter(fn: (r) => r[\"pid\"] == \"${npid}\")
    |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
    |> yield(name: \"mean\")"
else
  # Construction de la query sur l'ensemble des données
  query="data=from(bucket: \"telegraf_bucket\")
    |> range(start: ${t1}, stop: ${t2})
    |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
    |> yield(name: \"mean\")"
fi

# Ecriture de la query
echo $query > query

# Copie de la query dans le conteneur d'InfluxDB
sudo docker cp query influxdb:/query

# Exécution de la query dans le conteneur, et enregistrement de la sortie dans metrics_output
sudo docker exec -it influxdb sh -c 'influx query -f query -r' > metrics_output

echo "*************************************************"
echo "File metrics_output created"
echo "*************************************************"
head metrics_output
echo "*************************************************"
```

Format de sortie dans `metrics_output` :

```text
#group,false,false,true,true,false,false,true,true,true,true,true,true

#datatype,string,long,dateTime:RFC3339,dateTime:RFC3339,dateTime:RFC3339,double,string,string,string,string,string,string

#default,mean,,,,,,,,,,,

,result,table,_start,_stop,_time,_value,_field,_measurement,host,pattern,pid,process_name

,,0,2024-03-27T16:15:21.073488201Z,2024-03-27T16:15:57.890192557Z,2024-03-27T16:15:31Z,6.46,cpu_time_system,procstat,telegraf,.*,1601463,python3

,,0,2024-03-27T16:15:21.073488201Z,2024-03-27T16:15:57.890192557Z,2024-03-27T16:15:41Z,6.87,cpu_time_system,procstat,telegraf,.*,1601463,python3

,,0,2024-03-27T16:15:21.073488201Z,2024-03-27T16:15:57.890192557Z,2024-03-27T16:15:51Z,7.46,cpu_time_system,procstat,telegraf,.*,1601463,python3

[...]
```

## Suivre les performances énergétiques

Création du container Docker de l'application Home Assistant (dashboard spécifique optimisé pour Z-Wave): 

```bash
sudo docker run -d \
  --name homeassistant \
  --privileged \
  --restart=unless-stopped \
  -e TZ=Europe/Paris \
  -v ~/homeassistant:/config \
  -v /run/dbus:/run/dbus:ro \
  --network=host \
  ghcr.io/home-assistant/home-assistant:stable
```

Création du dossier contenant les configurations de Z-Wave: 

```bash
cd homeassistant
mkdir docker
mkdir docker/zwave-js
```

Récupération du nom du Network sur lequel a été installé telegraf :

```bash
sudo docker inspect telegraf -f '{{range $k, $v := .NetworkSettings.Networks}}{{printf "%s\n" $k}}{{end}}'
```

Récupération du nom du controleur USB :

```bash
dmesg | grep tty
```

Création du container Docker de l'application Z-Wave JS: 

```bash
sudo docker run -d \
  --network [TELEGRAF_NETWORK] \
  --restart=always \
  -p 8091:8091 \
  -p 3002:3000 \
  --device=[USB_CONTROLLER] \
  --name="zwave-js" \
  -e "TZ=Europe/Paris" \
  -v ~/homeassistant/docker/zwave-js:/usr/src/app/store zwavejs/zwavejs2mqtt:latest
```

Configuration de Z-Wave JS sur le port associé:

Pour configurer l'application Z-Wave JS afin de collecter les données de la prise intelligente, rendons nous à l'adresse `http://localhost:8091` et dans l'onglet `Smart Start`.

[Smart Start](smart-switch/smart-start.png)

Ajoutons les informations de notre périphérique Smart Switch, via le bouton `Add`, et ajoutons le code DSK de la prise intelligente (indiqué sur l'emballage) et activons tous les systèmes de sécurité.

[New entry](smart-switch/new-entry.png)

Une fois la prise intelligente connectée, configurons les paramètres de l'application. Dans l'onglet `Settings`, et la partie `Z-Wave`, ajoutons le nom du contrôleur USB identifié précedemment avant la création des conteneurs Docker. 

Vérifions que l'enregistrement des statistiques est activé.

[Enable statistics](smart-switch/enable-stats.png)

Enfin, dans la partie Home Assistant, ajoutons l'adresse IP du conteneur Z-Wave comme hôte. Celle-ci peut être identifiée via la commande `docker sudo docker inspect --format '{{ .NetworkSettings.IPAddress }}' zwave-js` dans le terminal. Nous pouvons aussi modifier le port si nous le souhaitons.

[Configure Home Assistant](smart-switch/config-homeassist.png)

Maintenant que l'application Z-Wave JS est configurée, rendons nous à l'adresse `http://localhost:8123` pour configurer l'application Home Assistant. Commençopns par créer un compte en suivant la procédure guidée à l'écran.
Une fois notre compte créé, ajoutons le périphérique d'intérêt.

Dans l'onglet `Settings`, rendons-nous sur la page `Devices & services`.
[Add device](smart-switch/add-device-ha.png)

Ajoutons une intégration Z-Wave en nous rendant sur la fonction `Add integration` et en sélectionnant `Z-Wave`. Il nous faut alors renseigner l'adresse configurée dans les paramètres de Z-Wave JS sous la forme `ws://[IP du conteneur zwave-js]:[port configuré]`.

Séléctionnons notre périphérique Smart Switch 7, et nous avons alors bien ajouté notre périphérique. Nous pouvons alors observer les premières acquisitions de données de la prise intelligente, et créer un dashboard si nous le souhaitons.
[Get first data](smart-switch/first-data.png)

### Connection à InfluxDB

```sh
pid_file /var/run/mosquitto.pid

persistence true
persistence_location /mosquitto/data/

log_dest file /mosquitto/log/mosquitto.log
log_dest stdout

password_file /mosquitto/config/mosquitto.passwd
allow_anonymous false
```

```bash
sudo docker run -d \
  --network [TELEGRAF_NETWORK] \
  --restart=always \
  -p 1883:1883 \
  --name="mosquitto" \
  -v ~/homeassistant/docker/mqtt:/mosquitto eclipse-mosquitto:1.6.15
```

```bash
sudo docker exec -it mosquitto sh
```

```bash
mosquitto_passwd -c mosquitto/config/mosquitto.passwd [user]
[password]
```

```squidconf
# # Read metrics from MQTT topic(s)
[[inputs.mqtt_consumer]]
#   ## Broker URLs for the MQTT server or cluster.  To connect to multiple
#   ## clusters or standalone servers, use a separate plugin instance.
#   ##   example: servers = ["tcp://localhost:1883"]
#   ##            servers = ["ssl://localhost:1883"]
#   ##            servers = ["ws://localhost:1883"]
   servers = ["tcp://mosquitto:1883"]
#
#   ## Topics that will be subscribed to.
   topics = [
     "zwave/Smart_switch_PC/50/0/value/65537",
     "zwave/Smart_switch_PC/50/0/value/66049",
     "zwave/Smart_switch_PC/50/0/value/66561",
     "zwave/Smart_switch_PC/50/0/value/66817",
   ]
#
#   ## The message topic will be stored in a tag specified by this value.  If set
#   ## to the empty string no topic tag will be created.
#   # topic_tag = "topic"
#
#   ## QoS policy for messages
#   ##   0 = at most once
#   ##   1 = at least once
#   ##   2 = exactly once
#   ##
#   ## When using a QoS of 1 or 2, you should enable persistent_session to allow
#   ## resuming unacknowledged messages.
#   # qos = 0
#
#   ## Connection timeout for initial connection in seconds
connection_timeout = "60s"
#
#   ## Max undelivered messages
#   ## This plugin uses tracking metrics, which ensure messages are read to
#   ## outputs before acknowledging them to the original broker to ensure data
#   ## is not lost. This option sets the maximum messages to read from the
#   ## broker that have not been written by an output.
#   ##
#   ## This value needs to be picked with awareness of the agent's
#   ## metric_batch_size value as well. Setting max undelivered messages too high
#   ## can result in a constant stream of data batches to the output. While
#   ## setting it too low may never flush the broker's messages.
#   # max_undelivered_messages = 1000
#
#   ## Persistent session disables clearing of the client session on connection.
#   ## In order for this option to work you must also set client_id to identify
#   ## the client.  To receive messages that arrived while the client is offline,
#   ## also set the qos option to 1 or 2 and don't forget to also set the QoS when
#   ## publishing. Finally, using a persistent session will use the initial
#   ## connection topics and not subscribe to any new topics even after
#   ## reconnecting or restarting without a change in client ID.
#   # persistent_session = false
#
#   ## If unset, a random client ID will be generated.
client_id = "telegraf"
#
#   ## Username and password to connect MQTT server.
username="********"
password="********"
#
#   ## Optional TLS Config
#   # tls_ca = "/etc/telegraf/ca.pem"
#   # tls_cert = "/etc/telegraf/cert.pem"
#   # tls_key = "/etc/telegraf/key.pem"
#   ## Use TLS but skip chain & host verification
#   # insecure_skip_verify = false
#
#   ## Client trace messages
#   ## When set to true, and debug mode enabled in the agent settings, the MQTT
#   ## client's messages are included in telegraf logs. These messages are very
#   ## noisey, but essential for debugging issues.
client_trace = true
#
#   ## Data format to consume.
#   ## Each data format has its own unique set of configuration options, read
#   ## more about them here:
#   ## https://github.com/influxdata/telegraf/blob/master/docs/DATA_FORMATS_INPUT.md
   data_format = "json"
   interval = "60s"
#
#   ## Enable extracting tag values from MQTT topics
#   ## _ denotes an ignored entry in the topic path
#   # [[inputs.mqtt_consumer.topic_parsing]]
#   #   topic = ""
#   #   measurement = ""
#   #   tags = ""
#   #   fields = ""
#   ## Value supported is int, float, unit
#   #   [[inputs.mqtt_consumer.topic.types]]
#   #      key = type
```

## Erreurs possibles

Dans le cas d'un arrêt complet du serveur, les adresses IP des conteneurs Docker peuvent alors être changées. Ceci ne devrait pas poser de problème pour l'ensemble du pipeline, excepté pour le software HomeAssistant.
Pour reconnecter HomeAssistant :

  * Récupérer l'adresse IP du conteneur de ZWave-JS UI via la commande ```sudo docker inspect --format '{{ .NetworkSettings.Networks.tig_default.IPAddress }}' zwave-js```

  * Réeffectuer la connection à HomeAssistant avec cette adresse IP comme présenté précédemment

Dans le cas où la prise connectée s'est éteinte, et que l'intervalle d'envoi de données à été modifié :

  * Se rendre sur l'interface de ZWave-JS UI `http://localhost:8091/` sur le panneau de contrôle (_Control panel_)

  * Sur le noeud associé à la prise connectée, dans l'onglet _Values_, ouvrir la rubrique _Configuration v1_

  * Modifier la valeur du paramètre _Automatic Reporting Interval_ (Attention, la valeur minimale est de 30 secondes)

[Pipeline final](smart-switch/pipeline.png)