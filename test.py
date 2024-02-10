import optuna

# Specify your storage URL
storage_url = "postgresql://optuna:optuna@optuna-db.melikbugraozcelik.com/optuna"  # Example for SQLite
# storage_url = "postgresql://user:password@localhost/dbname"  # Example for PostgreSQL, adjust as necessary

# Retrieve all study summaries
study_summaries = optuna.get_all_study_summaries(storage=storage_url)

# Delete each study
for study_summary in study_summaries:
    optuna.delete_study(study_name=study_summary.study_name, storage=storage_url)

print("All studies have been deleted.")
