from django.db import models

# Create your models here.

class Coffee(models.Model):
    id = models.AutoField(primary_key=True)
    product = models.CharField(max_length=30)
    price = models.SmallIntegerField()
    code = models.SmallIntegerField()
    

    def __str__(self):
        return self.product