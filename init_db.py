"""Initialize database tables and seed initial data."""

from server.orm import Base, engine, SessionLocal, TopicCluster

# Create all tables
print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("✓ Tables created successfully")

# Seed initial topic clusters
print("\nSeeding initial topic clusters...")
db = SessionLocal()

initial_clusters = [
    {"id": 1, "label": "Registration & Enrollment"},
    {"id": 2, "label": "Billing & Financial Aid"},
    {"id": 3, "label": "IT & Technology"},
    {"id": 4, "label": "Housing & Residential Life"},
    {"id": 5, "label": "Library & Academic Resources"},
]

for cluster_data in initial_clusters:
    existing = db.query(TopicCluster).filter(TopicCluster.id == cluster_data["id"]).first()
    if not existing:
        cluster = TopicCluster(**cluster_data)
        db.add(cluster)
        print(f"  + Added cluster: {cluster_data['label']}")
    else:
        print(f"  - Cluster already exists: {cluster_data['label']}")

db.commit()
db.close()

print("\n✓ Database initialization complete!")
