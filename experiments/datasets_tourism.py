import mlrun
import pandas as pd

project_tourism = mlrun.get_or_create_project("overtourism")

contamezzi_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/contamezzi.parquet.parquet").as_df()
manifestazioni_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/manifestazioni.parquet.parquet").as_df()
meteotrentino_bollettino_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/meteotrentino_bollettino.parquet.parquet").as_df()
contamezzi_descrizione_sensore_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/contamezzi_descrizione_sensore.parquet.parquet").as_df()
contapersone_passaggi_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/contapersone_passaggi.parquet.parquet").as_df()
contapersone_presenze_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/contapersone_presenze.parquet.parquet").as_df()
statistiche_parcheggi_molveno_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/statistiche_parcheggi_molveno.parquet.parquet").as_df()

movimento_turistico_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/movimento_turistico.parquet.parquet").as_df()
extra_strutture_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/extra_strutture.parquet.parquet").as_df()
survey_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/survey.parquet.parquet").as_df()
vodafone_aree_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/vodafone_aree.parquet.parquet").as_df()
vodafone_attendences_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/vodafone_attendences.parquet.parquet").as_df()
vodafone_attendences_STR_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/vodafone_attendences_STR.parquet.parquet").as_df()

movimento_turistico_molveno_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/movimento_turistico_molveno.parquet.parquet").as_df()
dati_pioggia_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/dati_pioggia.parquet.parquet").as_df()

depuratore_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/depuratore.parquet.parquet").as_df()
parcheggi_posti_molveno_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/parcheggi_posti.parquet.parquet").as_df()
ristoranti_spiaggia_df = mlrun.get_dataitem("s3://datalake/projects/overtourism/artifacts/ristoranti_spiaggia.parquet.parquet").as_df()

# tgp_card data


