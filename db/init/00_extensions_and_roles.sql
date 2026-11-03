-- Runs once when the Postgres container is first created.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS ltree;

-- The API connects as this role. It is not a superuser and does not own the
-- tables, so row-level security always applies to it.
CREATE ROLE secondbrain_app LOGIN PASSWORD 'dev_app_password';
GRANT CONNECT ON DATABASE secondbrain TO secondbrain_app;
GRANT USAGE ON SCHEMA public TO secondbrain_app;

-- Separate database for the test suite, so tests never touch demo data.
CREATE DATABASE secondbrain_test;
GRANT CONNECT ON DATABASE secondbrain_test TO secondbrain_app;
\connect secondbrain_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS ltree;
GRANT USAGE ON SCHEMA public TO secondbrain_app;
