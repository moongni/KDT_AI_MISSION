# Generated by Django 4.2 on 2023-04-19 08:39

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Coffee",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("product", models.CharField(max_length=30)),
                ("price", models.SmallIntegerField()),
                ("code", models.SmallIntegerField()),
            ],
        ),
    ]
