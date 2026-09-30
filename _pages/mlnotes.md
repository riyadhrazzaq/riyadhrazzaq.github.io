---
layout: page
title: ml notes
permalink: /mlnotes/
description: short notes on machine learning building blocks
nav: true
nav_order: 1.5
---

<div class="table-responsive">
  <table class="table table-sm table-borderless">
    {% for post in site.tags.mlnote %}
      <tr>
        <th scope="row">{{ post.date | date: '%b %d, %Y' }}</th>
        <td>
          {% if post.redirect == blank %}
            <a class="post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
          {% elsif post.redirect contains '://' %}
            <a class="post-link" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
          {% else %}
            <a class="post-link" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
          {% endif %}
        </td>
      </tr>
    {% endfor %}
  </table>
</div>
