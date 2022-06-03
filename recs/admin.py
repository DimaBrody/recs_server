from django.contrib import admin
from .models import Book, BookCB


class BookAdmin(admin.ModelAdmin):
    list_display = ("title", "rating")


admin.site.register(Book)
admin.site.register(BookCB)
