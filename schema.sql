create table repo (
    pr_number integer not null primary key,
    name text not null,
    embedding vector(512)
)