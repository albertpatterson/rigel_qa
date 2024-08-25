
import re
import os
import datetime



def get_classes(el):
  if el.has_attr('class'):
    return el['class']
  return []

def is_stop_el(el):
  if el is None:
    return False
  if isinstance(el, str):
    return False
  if not el.has_attr('class'):
    return False

  out = 'sharedaddy' in el['class']
  return out
  

def should_stop(el):
  if el is None:
    return True
  if isinstance(el, str):
    return False
  if is_stop_el(el):
    return True

  return False

def is_interesting(el):
  if el.name == 'script':
    return False

  return True

def get_post_content(div):
  all_content_parts = []
  el = div.select('div.post-info')[0].next_sibling

  while not should_stop(el):
    if is_interesting(el):
      if isinstance(el, str):
        text = el
      else:
        text = el.get_text(separator='\n', strip=True)


      all_content_parts.append(text)
    el = el.next_sibling
  return '\n'.join(all_content_parts)


def get_defaulted(getter, el, default=""):
  try:
    return getter(el)
  except Exception as e:
    print('ERROR')
    print(e)
    return default

def get_title(div):
  return div.find_next('div').find_next('h2').text

def get_link(div):
  return div.find_next('div').find_next('a')['href']

def get_date(div):
  return div.select_one('span.time').text

def blog_date_string_to_date(date_string):
  """Converts a string to a datetime object."""
  date_format = "%B %d, %Y"
  date_obj = datetime.datetime.strptime(date_string, date_format)
  return date_obj

def get_post_data(div):
  title = get_defaulted(get_title, div, "")
  link = get_defaulted(get_link, div, "")
  classes = get_defaulted(get_classes, div, [])
  datestr = get_defaulted(get_date, div, None)
  date = blog_date_string_to_date(datestr) if datestr is not None else None
  content = get_defaulted(get_post_content, div, "")
  return dict(title=title, link=link, classes=classes, date=date, content=content)